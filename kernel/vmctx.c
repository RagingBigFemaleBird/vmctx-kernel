// SPDX-License-Identifier: GPL-2.0
/*
 * vmctx — native "VM context" for Linux (avm.md project).
 *
 * Core: makes a VM context a first-class process. A task that calls
 * vmctx_run(2) becomes a VM context — it keeps its pid, /proc entry,
 * signal handling and scheduler treatment, but its execution is carried by
 * a hardware guest via the registered backend.
 *
 * Guest entry happens in the context-switch / return-to-user path: the
 * generic exit_to_user_mode_loop() calls vmctx_guest_step() for a VM-context
 * task, so the guest is (re)entered whenever the scheduler switches the task
 * in and heads back to user space, and a guest VMEXIT that sets need_resched
 * is followed by that same loop's schedule() — VM enter/exit at the switch
 * boundary. kill/wait/nice/ps therefore work because it is an ordinary task.
 *
 * The backend (AMD SVM / Intel VMX) is registered by a module and does the
 * actual VMRUN; see struct vmctx_backend.
 */
#include <linux/mman.h>
#include <linux/sched/debug.h>
#include <linux/vmctx.h>
#include <linux/mmu_notifier.h>

/* Defined with the rest of the secondary-TLB work, used from the run loop. */
struct vmctx_task;
static void vmctx_mn_attach(struct vmctx_task *vc);
static void vmctx_mn_detach(struct vmctx_task *vc);
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/module.h>
#include <linux/syscalls.h>
#include <linux/uaccess.h>
#include <linux/futex.h>
#include <linux/export.h>
#include <linux/ptrace.h>
#include <linux/sched/task.h>
#include <linux/sched/mm.h>
#include <linux/wait.h>
#include <linux/pid.h>
#include <linux/ptrace.h>
#include <linux/delay.h>
#include <linux/mm.h>
/* arch_do_signal_or_restart(): a forwarded syscall ends where any syscall
 * ends, in the architecture's return-to-user work. See
 * vmctx_do_monitor_syscall(). */
#include <linux/irq-entry-common.h>
/* The address-space-wide hand-over; see vmctx_ctl_takeobj(). */
#include <linux/pagemap.h>
#include <linux/shmem_fs.h>
#include <linux/rmap.h>	/* rmap_walk()/page_vma_mapped_walk(): object-wide wrprotect */
#include <linux/swap.h>
#include <linux/fs.h>
/* Splitting a large folio so the precheck can answer for it; dump_page(). */
#include <linux/huge_mm.h>
#include <linux/mmdebug.h>
#include <linux/hash.h>	/* hash_long(): the per-page allocation stamp */
#include <asm/fsgsbase.h>

/* May a page change machines right now? Defined beside vmctx_ctl_takeobj(). */
static long vmctx_folio_claim(struct folio *folio, int extra_refs);
static void vmctx_folio_release(struct folio *folio, int extra_refs);
static long vmctx_folio_precheck(struct folio *folio, int max_maps,
				 int extra_refs, const char *what);

/*
 * Every way a take can end, counted, and readable while the system runs:
 *
 *   /sys/module/kernel/parameters/vmctx_take_*
 *
 * They exist because the refusal that mattered most was invisible. The one
 * instrument that reported it -- the "page busy #%ld" line in
 * vmctx_folio_claim() -- prints only its first eight since boot and then one in
 * 1024, so a run that refused a hundred hand-overs after the boot's first eight
 * printed nothing at all, and "zero refusals in dmesg" meant nothing. A counter
 * that is always incremented and always readable cannot be rate-limited into
 * agreeing with a hypothesis (PRINCIPLES.md §7: absence of evidence is evidence
 * only if the check can fire).
 *
 * vmctx_take_lost is the one that has to stay zero: it counts takes refused
 * *after* the page tables were destroyed, which is the refusal that costs the
 * page.
 */
static long vmctx_take_calls;
static long vmctx_take_absent;		/* nothing here; not an error */
static long vmctx_take_hold;		/* refused: an in-flight syscall holds it */
static long vmctx_take_refused;		/* refused BEFORE the zap: page intact */
static bool vmctx_page_present_in(struct mm_struct *mm, unsigned long addr);
static long vmctx_syscall_pages_lost;	/* written pages gone before the call ended */
static long vmctx_syscall_pages_checked;/* ...out of how many, the denominator */
static long vmctx_syscall_calls_checked;/* ...over how many forwarded calls */
static long vmctx_syscall_calls_lost;	/* ...and how many calls that was */
static long vmctx_take_lost;		/* refused AFTER the zap: page destroyed */
static long vmctx_take_regained;	/* post-zap claim that a retry won back */
static long vmctx_take_large;		/* large folios met by a take */
static long vmctx_precheck_large;	/* large folios the precheck cannot answer for */
static long vmctx_take_split;		/* large folios split into order-0 ones */
static long vmctx_take_split_fail;	/* a split that would not happen: -EBUSY */
static long vmctx_take_drained;		/* a refusal a full LRU drain resolved */
static long vmctx_take_zeropage;	/* takes that found the shared zero page */
static long vmctx_takeobj_refused;	/* takeobj refused before the unmap */
static long vmctx_takeobj_lost;		/* takeobj refused after it */

/*
 * The fix this file exists to carry, with a switch so it can be measured
 * against itself on one boot rather than against a memory of another kernel.
 * 1 (default): ask whether the page may move before destroying anything.
 * 0: the old order -- zap, then ask -- kept only as the control.
 */
static bool vmctx_take_claim_first = true;

/*
 * How many times a claim that lost the race after the zap is asked again before
 * the page is given up for lost. 0 reproduces the old kernel exactly, which is
 * the only way to check that the instrument measuring all this can fire at all
 * (PRINCIPLES.md §7).
 */
static int vmctx_take_retries = 4;

/*
 * A large folio cannot be prechecked -- folio_mapcount() of one does not say
 * how many mappings a 4 KiB zap removes -- so it is made into folios that can
 * be: split to order 0 before the question is asked. 1 (default): split.
 * 0: the old behaviour, which was to count the folio and hand it to the claim,
 * where a refusal costs the page. Kept as the control.
 */
static bool vmctx_take_split_large = true;

/*
 * Drain every CPU's folio batches on any reference mismatch, not only when the
 * folio is off the LRU.
 *
 * The precheck used to drain only for !folio_test_lru(), on the reasoning that
 * the extra reference is the LRU-*add* batch's slot. That is one batch of
 * several: folio_activate(), lru_deactivate_file(), lru_lazyfree() and the
 * mlock batches each hold a reference to a folio that is already on the LRU,
 * so "lru=1" excluded exactly the batches that can hold a reference while
 * lru=1. 0 restores the narrow drain, as the control.
 */
static bool vmctx_take_drain_always = true;

/*
 * A take that finds the shared zero page answers "nothing here" instead of
 * refusing it. Tried, measured, and left OFF: see vmctx_ctl_take(). 0
 * (default): refuse it, and say in the log that the refusal can never succeed.
 */
static bool vmctx_take_zeropage_absent;

/*
 * How many refused prechecks get a dump_page() naming what actually holds the
 * folio. Counted down, so it costs nothing after the first few; write it again
 * to arm the instrument for the next run. 0 turns it off.
 */
static int vmctx_take_dump = 8;

core_param(vmctx_take_claim_first, vmctx_take_claim_first, bool, 0644);
core_param(vmctx_take_retries, vmctx_take_retries, int, 0644);
core_param(vmctx_take_split_large, vmctx_take_split_large, bool, 0644);
core_param(vmctx_take_drain_always, vmctx_take_drain_always, bool, 0644);
core_param(vmctx_take_dump, vmctx_take_dump, int, 0644);
core_param(vmctx_take_split, vmctx_take_split, long, 0644);
core_param(vmctx_take_split_fail, vmctx_take_split_fail, long, 0644);
core_param(vmctx_take_drained, vmctx_take_drained, long, 0644);
core_param(vmctx_take_zeropage, vmctx_take_zeropage, long, 0644);
core_param(vmctx_take_zeropage_absent, vmctx_take_zeropage_absent, bool, 0644);
core_param(vmctx_take_calls, vmctx_take_calls, long, 0644);
core_param(vmctx_take_absent, vmctx_take_absent, long, 0644);
core_param(vmctx_take_hold, vmctx_take_hold, long, 0644);
core_param(vmctx_take_refused, vmctx_take_refused, long, 0644);
core_param(vmctx_syscall_pages_lost, vmctx_syscall_pages_lost, long, 0644);
core_param(vmctx_syscall_pages_checked, vmctx_syscall_pages_checked, long, 0644);
core_param(vmctx_syscall_calls_checked, vmctx_syscall_calls_checked, long, 0644);
core_param(vmctx_syscall_calls_lost, vmctx_syscall_calls_lost, long, 0644);
core_param(vmctx_take_lost, vmctx_take_lost, long, 0644);
core_param(vmctx_take_regained, vmctx_take_regained, long, 0644);
core_param(vmctx_take_large, vmctx_take_large, long, 0644);
core_param(vmctx_precheck_large, vmctx_precheck_large, long, 0644);
core_param(vmctx_takeobj_refused, vmctx_takeobj_refused, long, 0644);
core_param(vmctx_takeobj_lost, vmctx_takeobj_lost, long, 0644);

/*
 * How VMCTX_CTL_PROTECTOBJ write-protects each mapping of the folio. A runtime
 * A/B knob (/sys/module/kernel/parameters/vmctx_protobj_mode) to separate two
 * entangled effects: clearing the dirty bit, and the not-present window of
 * ptep_clear_flush(). folio_mkclean() does both and fixes hx2 but breaks pg1;
 * ptep_set_wrprotect() does neither and does the opposite.
 *   0 = ptep_set_wrprotect + flush     (keep dirty, no window)  [pg1-safe]
 *   1 = ptep_clear_flush + wrprotect   (keep dirty, with window)
 *   2 = ptep_clear_flush + wrprotect+mkclean (clear dirty, with window) [==folio_mkclean]
 *   3 = set_pte wrprotect+mkclean + flush    (clear dirty, no window)
 */
static int vmctx_protobj_mode;
core_param(vmctx_protobj_mode, vmctx_protobj_mode, int, 0644);
static long vmctx_protobj_prot;		/* mappings protected, for control */
core_param(vmctx_protobj_prot, vmctx_protobj_prot, long, 0644);

/*
 * VMEXIT-scan lost-write detector. The guest's store is the one event nothing
 * observes -- a bare hardware write to the shared folio, not a syscall, fault or
 * lock -- so every userspace instrument is circular (it can only compare against
 * what a take already read). This closes that: on every VMEXIT (the instant
 * after the guest ran) sample the watched page's eight writer-slot longs
 * (offsets 64..127) into a high-water-mark taken from the guest's OWN stores;
 * then any take/serve that produces a value below the mark reverted a store, and
 * we name the op that did it. Set vmctx_watch_gva to the page base (guest VA) to
 * arm it; 0 is off and costs nothing.
 */
static unsigned long vmctx_watch_gva;
core_param(vmctx_watch_gva, vmctx_watch_gva, ulong, 0644);
static unsigned long vmctx_watch_hwm[8];
static DEFINE_SPINLOCK(vmctx_watch_lock);
static long vmctx_watch_back;		/* backward events seen (the lost writes) */
core_param(vmctx_watch_back, vmctx_watch_back, long, 0644);
static long vmctx_watch_samples;	/* VMEXIT samples taken, for control */
core_param(vmctx_watch_samples, vmctx_watch_samples, long, 0644);

/* On VMEXIT (current is the guest task; the watched page is in its mm): raise
 * the mark from the guest's actual stores. Exported for the backend module. */
static long vmctx_watch_calls;		/* every call (debug) */
core_param(vmctx_watch_calls, vmctx_watch_calls, long, 0644);
static long vmctx_watch_efault;		/* guest-view copy faulted (page absent) */
core_param(vmctx_watch_efault, vmctx_watch_efault, long, 0644);
static long vmctx_watch_diverge;	/* guest PTE view != object view: the loss */
core_param(vmctx_watch_diverge, vmctx_watch_diverge, long, 0644);

/* Read the OBJECT folio (memfd) at the watched offset -- not the guest's PTE,
 * the object itself. Returns true and fills ov[8] if the object has the page. */
static bool vmctx_watch_object(unsigned long gva, unsigned long *ov,
			       unsigned long *obj_pfn)
{
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma;
	struct folio *folio;
	pgoff_t index;
	bool ok = false;
	void *k;

	*obj_pfn = 0;
	if (!mm || !mmap_read_trylock(mm))
		return false;
	vma = vma_lookup(mm, gva);
	if (vma && vma->vm_file && (vma->vm_flags & VM_SHARED)) {
		index = linear_page_index(vma, gva);
		folio = filemap_get_folio(vma->vm_file->f_mapping, index);
		if (!IS_ERR(folio)) {
			if (!folio_test_large(folio)) {
				k = kmap_local_folio(folio, 0);
				memcpy(ov, (char *)k + 64, 64);
				kunmap_local(k);
				*obj_pfn = folio_pfn(folio);
				ok = true;
			}
			folio_put(folio);
		}
	}
	mmap_read_unlock(mm);
	return ok;
}

/*
 * The guest just ran. Read what the guest's OWN page tables map at the watched
 * slots (gv), and what the backing OBJECT holds at the same slots (ov). If they
 * differ, the guest is reading a page the object no longer has -- an orphaned or
 * stale mapping -- which is a lost write caught at its source: the guest sees
 * gv, the take/serve sees ov, and one of them is wrong. This is the only
 * non-circular check, because gv comes from the guest and ov from the object.
 */
static long vmctx_watch_samefolio;	/* diverged but SAME folio = my race */
core_param(vmctx_watch_samefolio, vmctx_watch_samefolio, long, 0644);
static long vmctx_watch_orphan;		/* diverged AND different folio = real */
core_param(vmctx_watch_orphan, vmctx_watch_orphan, long, 0644);
/*
 * The fast-writer version of the split above, which is what hx2 needs: any
 * sample where the guest's value is BELOW the take-seeded high-water mark is a
 * stale read (the guest reads a value already handed over), regardless of
 * whether it is stable -- so the fast writer no longer hides it. Then split by
 * the physical page the guest maps: STALE_SAME means the current folio itself
 * holds an old value (a stale serve/write into it); STALE_ORPHAN means the
 * guest maps a DIFFERENT folio than the object (it is writing/reading a page the
 * coherence no longer tracks -- the flush failed to re-fault it onto the object).
 */
static long vmctx_watch_stale_same;
core_param(vmctx_watch_stale_same, vmctx_watch_stale_same, long, 0644);
static long vmctx_watch_stale_orphan;
core_param(vmctx_watch_stale_orphan, vmctx_watch_stale_orphan, long, 0644);

void vmctx_watch_sample(void)
{
	unsigned long gv1[8], gv2[8], ov[8], obj_pfn = 0, gpfn = 0;
	struct page *pg = NULL;
	int i, differ = 0, stale = 0;

	vmctx_watch_calls++;
	if (!vmctx_watch_gva)
		return;
	if (copy_from_user_nofault(gv1, (void __user *)(vmctx_watch_gva + 64),
				   sizeof(gv1))) {
		vmctx_watch_efault++;
		return;
	}
	vmctx_watch_samples++;
	if (!vmctx_watch_object(vmctx_watch_gva, ov, &obj_pfn))
		return;
	/* Re-read the guest AFTER the object read. If the guest value moved,
	 * this is just the folio being written between my two reads (a monotonic
	 * writer always makes the later read larger) -- not a stale mapping. Only
	 * a STABLE guest value that still differs from the object is real. */
	if (copy_from_user_nofault(gv2, (void __user *)(vmctx_watch_gva + 64),
				   sizeof(gv2)))
		return;
	/* Which physical page does the guest actually map here? Needed by both
	 * the stable-divergence test and the stale-vs-mark test below. */
	if (get_user_pages_fast_only(vmctx_watch_gva, 1, 0, &pg) == 1 && pg) {
		gpfn = page_to_pfn(pg);
		put_page(pg);
	}
	for (i = 0; i < 8; i++) {
		if (gv1[i] != ov[i] && gv1[i] == gv2[i])
			differ = 1;
		/* Below the take-seeded mark = a stale read, fast writer and all. */
		if (gv1[i] >= 1 && gv1[i] < (1UL << 32) &&
		    vmctx_watch_hwm[i] >= 1 && gv1[i] < vmctx_watch_hwm[i])
			stale = 1;
	}
	if (stale) {
		if (gpfn && obj_pfn && gpfn == obj_pfn)
			vmctx_watch_stale_same++;
		else
			vmctx_watch_stale_orphan++;
	}
	if (!differ)
		return;
	vmctx_watch_diverge++;
	if (gpfn && obj_pfn && gpfn == obj_pfn) {
		vmctx_watch_samefolio++;	/* same folio: instrument artifact */
	} else {
		vmctx_watch_orphan++;
		pr_warn_ratelimited("vmctx: ORPHAN 0x%lx: guest stable %lu on pfn %lx, object %lu on pfn %lx -- guest maps a different (stale) folio\n",
				    vmctx_watch_gva, gv1[0], gpfn, ov[0], obj_pfn);
	}
}
EXPORT_SYMBOL_GPL(vmctx_watch_sample);

/* A take/serve that has `page` bytes for guest VA `addr`: flag any slot that
 * has gone below what the guest was last seen to have stored. */
static void vmctx_watch_check(unsigned long addr, const void *page,
			      const char *op)
{
	const unsigned long *sl = (const unsigned long *)
		((const char *)page + 64);
	int i;

	if (!vmctx_watch_gva || (addr & PAGE_MASK) != vmctx_watch_gva)
		return;
	spin_lock(&vmctx_watch_lock);
	for (i = 0; i < 8; i++) {
		unsigned long hw = vmctx_watch_hwm[i], n = sl[i];

		/* Monotonic counters only, so pointers/text/zero cannot poison
		 * the mark. This runs at TAKE/TAKEOBJ, which read the LIVE folio
		 * under the folio lock -- the one place the kernel sees the
		 * guest's true current value -- so the mark it seeds is the real
		 * high-water of what has changed hands, not a lagging VMEXIT
		 * sample (which never wrote this array; it was dead). */
		if (n < 1 || n >= (1UL << 32))
			continue;
		if (hw >= 1 && n < hw) {
			vmctx_watch_back++;
			pr_warn_ratelimited("vmctx: LOSTWRITE via %s 0x%lx slot %d: produced %lu < guest-store hwm %lu -- this op reverted the guest's write\n",
					    op, vmctx_watch_gva, i * 8, n, hw);
		}
		if (n > hw)
			vmctx_watch_hwm[i] = n;
	}
	spin_unlock(&vmctx_watch_lock);
}

/*
 * The counterpart check: a value the guest is about to READ BACK after a
 * fault-serve made its page present again. If it is below the high-water-mark
 * of what has changed hands (seeded above from the live folio at every take),
 * the serve handed the guest a copy older than a value already handed over --
 * the stale read hx2 measures, and the transient the VMEXIT sampler misses
 * because it is corrected before the next sample. Seeds nothing: the served
 * value is exactly what may be stale, so it must never raise the mark.
 */
static long vmctx_watch_regress;	/* guest read below the handed-over mark */
core_param(vmctx_watch_regress, vmctx_watch_regress, long, 0644);
static unsigned long vmctx_watch_regress_worst;	/* worst backward gap seen */
core_param(vmctx_watch_regress_worst, vmctx_watch_regress_worst, ulong, 0644);

unsigned long vmctx_watch_page(void)
{
	return vmctx_watch_gva;
}
EXPORT_SYMBOL_GPL(vmctx_watch_page);

void vmctx_watch_served(unsigned long gva, const unsigned long *sl8)
{
	int i;

	if (!vmctx_watch_gva || (gva & PAGE_MASK) != vmctx_watch_gva)
		return;
	spin_lock(&vmctx_watch_lock);
	for (i = 0; i < 8; i++) {
		unsigned long hw = vmctx_watch_hwm[i], n = sl8[i];

		if (hw >= 1 && hw < (1UL << 32) && n >= 1 && n < hw) {
			unsigned long gap = hw - n;

			vmctx_watch_regress++;
			if (gap > vmctx_watch_regress_worst)
				vmctx_watch_regress_worst = gap;
			if (vmctx_watch_regress <= 16)
				pr_warn("vmctx: STALE-READ 0x%lx slot %d: guest reads %lu < handed-over hwm %lu (gap %lu) -- served a copy older than what changed hands\n",
					gva, i * 8, n, hw, gap);
		}
	}
	spin_unlock(&vmctx_watch_lock);
}
EXPORT_SYMBOL_GPL(vmctx_watch_served);

/*
 * Anonymous faults taken on somebody else's address space. See
 * vmctx_foreign_anon_fault(): vmctx_all counts every one on the machine, and
 * vmctx_foreign_ctx counts the ones where that address space belongs to a live
 * VM context -- a page of a context supplied by this kernel with its monitor
 * never asked.
 */
static long vmctx_foreign_all;
static long vmctx_foreign_ctx;
/*
 * ...and of those, the ones that were READS.
 *
 * A write is how a monitor installs a page it already holds: it pokes the
 * bytes in, the kernel gives it an empty page a moment before, and the poke
 * covers it. That is noisy but harmless. A read is not: nobody is about to
 * write real bytes over it, so what the reader gets -- and what the context is
 * left holding for ever after -- is a page this kernel invented, for an address
 * whose contents live on another machine.
 */
static long vmctx_foreign_read;
/* Reads refused rather than answered with a page this kernel made up. */
static long vmctx_foreign_refused;
static bool vmctx_foreign_refuse = true;
core_param(vmctx_foreign_refused, vmctx_foreign_refused, long, 0644);
core_param(vmctx_foreign_refuse, vmctx_foreign_refuse, bool, 0644);
/*
 * Writes to a present-but-read-only page of a context's address space, by kind.
 *
 *   own      the context itself: copy-on-write after a guest fork, or the
 *            shared zero page being written for the first time. The bytes are
 *            here, so nothing is invented -- but nothing is recorded either,
 *            and a context whose page the other machine has since written would
 *            get this machine's copy without anyone being asked.
 *   foreign  somebody else writing it, which is a monitor's POKE breaking a
 *            copy-on-write.
 *   shared   of those, mappings the write only dirties (the destination's
 *            memory is MAP_SHARED from its backing object, so this is expected
 *            and harmless); private ones are the copy-on-write case.
 */
static long vmctx_wp_own, vmctx_wp_foreign, vmctx_wp_shared, vmctx_wp_filed;
/*
 * ...and the one that would invent memory: a write to the shared zero page.
 * There is nothing to copy, so what the write gets is a freshly zeroed page for
 * an address whose contents may be on another machine. Refused for the same
 * reason a foreign read is (vmctx_foreign_anon_fault): there is nothing to ask
 * -- do_wp_page() holds the page-table lock -- and a page this kernel made up
 * is not an answer. vmctx_wp_zero_refuse=0 restores the old behaviour.
 */
static long vmctx_wp_zero, vmctx_wp_zero_refused, vmctx_wp_zero_own;
static bool vmctx_wp_zero_refuse = true;
core_param(vmctx_wp_zero, vmctx_wp_zero, long, 0644);
core_param(vmctx_wp_zero_refused, vmctx_wp_zero_refused, long, 0644);
core_param(vmctx_wp_zero_own, vmctx_wp_zero_own, long, 0644);
core_param(vmctx_wp_zero_refuse, vmctx_wp_zero_refuse, bool, 0644);
core_param(vmctx_wp_own, vmctx_wp_own, long, 0644);
core_param(vmctx_wp_foreign, vmctx_wp_foreign, long, 0644);
core_param(vmctx_wp_shared, vmctx_wp_shared, long, 0644);
core_param(vmctx_wp_filed, vmctx_wp_filed, long, 0644);
core_param(vmctx_foreign_all, vmctx_foreign_all, long, 0644);
core_param(vmctx_foreign_ctx, vmctx_foreign_ctx, long, 0644);
core_param(vmctx_foreign_read, vmctx_foreign_read, long, 0644);

/*
 * Address spaces that belong to a VM context, so a fault taken on one by any
 * task can be recognised. Registered when a context starts holding a program
 * and forgotten when it stops; a handful at a time, and only consulted for a
 * fault on a foreign mm, which is rare.
 */
/*
 * The address spaces that belong to a context, and how many contexts each one
 * has. Every hook in this file asks this table first, so what is in it decides
 * whether a fault on a page is asked about or silently invented.
 *
 * THE COUNT IS THE WHOLE POINT, and it was missing. This was a set: note()
 * deduplicated, so four threads of one program filled one slot, and forget()
 * -- called from vmctx_task_exit(), once per exiting CONTEXT -- cleared that
 * slot the first time ANY of them exited. From that instant the address space
 * was not a context's as far as this file was concerned, while its other
 * threads were still running in it. Every hook fell through:
 *
 *   do_anonymous_page()  invented pages for addresses the other machine owns
 *   do_wp_page()         stopped being counted, let alone refused
 *   vmctx_install_single() went false, so THP and large-folio batch installs
 *                        came back and populated whole ranges of NEIGHBOURS
 *                        that nobody faulted on
 *   vmctx_fault_seen()   stopped counting -- so the instrument that was
 *                        supposed to notice went blind at the same instant,
 *                        which is why "every fault is reported" measured clean
 *
 * tests/ws1 is exactly this shape: four threads, each returns, then main
 * touches .bss on its way through exit(). Once the first worker exits, main's
 * memory is unhooked for the rest of the run.
 *
 * So the slot is refcounted, incremented and decremented on precisely the
 * events that already maintain vmctx_nr_active -- vc->active going 0->1 and
 * 1->0 -- which is what makes the pairing exact rather than hopeful.
 */
#define VMCTX_MM_SLOTS 64
struct vmctx_mm_slot {
	struct mm_struct *mm;
	int n;			/* active contexts in this address space */
};
static struct vmctx_mm_slot vmctx_mms[VMCTX_MM_SLOTS];
static DEFINE_SPINLOCK(vmctx_mm_lock);

/*
 * How often the old rule would have unhooked an address space that still had
 * contexts running in it. Counted in BOTH arms, because it is a fact about the
 * program rather than about the fix: it says how often the bug had the chance
 * to fire, which is the denominator every measurement below needs.
 */
static long vmctx_mm_early_drop;
static long vmctx_mm_overflow;	/* no free slot: unhooked, silently, until now */
static long vmctx_mm_live;	/* address spaces currently in the table       */
static bool vmctx_mm_refcount = true;
core_param(vmctx_mm_early_drop, vmctx_mm_early_drop, long, 0644);
core_param(vmctx_mm_overflow, vmctx_mm_overflow, long, 0644);
core_param(vmctx_mm_live, vmctx_mm_live, long, 0444);
core_param(vmctx_mm_refcount, vmctx_mm_refcount, bool, 0644);

static void vmctx_mm_note(struct mm_struct *mm)
{
	unsigned long flags;
	int i, free = -1;

	if (!mm)
		return;
	spin_lock_irqsave(&vmctx_mm_lock, flags);
	for (i = 0; i < VMCTX_MM_SLOTS; i++) {
		if (vmctx_mms[i].mm == mm) {
			vmctx_mms[i].n++;
			goto out;
		}
		if (!vmctx_mms[i].mm && free < 0)
			free = i;
	}
	if (free >= 0) {
		vmctx_mms[free].mm = mm;
		vmctx_mms[free].n  = 1;
		vmctx_mm_live++;
	} else {
		/*
		 * Nowhere to put it, and until this counter existed that was
		 * silent: the address space simply was not a context's, so
		 * every hook fell through for the whole of its life. Said out
		 * loud, because a run that hits this is not measuring what it
		 * thinks it is.
		 */
		vmctx_mm_overflow++;
		pr_warn_ratelimited("vmctx: no free mm slot (%d in use); this address space's faults will NOT be asked about\n",
				    VMCTX_MM_SLOTS);
	}
out:
	spin_unlock_irqrestore(&vmctx_mm_lock, flags);
}

static void vmctx_mm_forget(struct mm_struct *mm)
{
	unsigned long flags;
	int i;

	if (!mm)
		return;
	spin_lock_irqsave(&vmctx_mm_lock, flags);
	for (i = 0; i < VMCTX_MM_SLOTS; i++)
		if (vmctx_mms[i].mm == mm) {
			if (--vmctx_mms[i].n > 0) {
				/*
				 * Other contexts are still running in this
				 * address space. The old rule dropped it here.
				 */
				vmctx_mm_early_drop++;
				if (vmctx_mm_refcount)
					break;
			}
			vmctx_mms[i].mm = NULL;
			vmctx_mms[i].n  = 0;
			vmctx_mm_live--;
			break;
		}
	spin_unlock_irqrestore(&vmctx_mm_lock, flags);
}

bool vmctx_mm_is_context(struct mm_struct *mm)
{
	unsigned long flags;
	int i;
	bool theirs = false;

	if (!mm)
		return false;
	spin_lock_irqsave(&vmctx_mm_lock, flags);
	for (i = 0; i < VMCTX_MM_SLOTS; i++)
		if (vmctx_mms[i].mm == mm) {
			theirs = true;
			break;
		}
	spin_unlock_irqrestore(&vmctx_mm_lock, flags);
	return theirs;
}
EXPORT_SYMBOL_GPL(vmctx_mm_is_context);

/*
 * The mm_forbids_zeropage() hook -- see the comment at its definition in
 * <linux/mm.h>. Called from every anonymous read fault on the machine, so
 * it must cost nothing where vmctx is idle and no lock anywhere: the
 * vmctx_mm_live read is the whole cost on an ordinary box. The slot scan
 * is deliberately lockless -- a false negative (mm adopted mid-scan) is
 * today's behavior arriving one adoption earlier, and a false positive
 * (mm just released) costs one ordinary zeroed folio. Both are benign;
 * the fault path pays no spinlock. The counter is racy for the same
 * reason and is a diagnostic, not an accounting.
 */
static long vmctx_zeropage_forbidden;
core_param(vmctx_zeropage_forbidden, vmctx_zeropage_forbidden, long, 0444);

bool vmctx_mm_forbids_zeropage(struct mm_struct *mm)
{
	int i;

	if (likely(!READ_ONCE(vmctx_mm_live)) || !mm)
		return false;
	for (i = 0; i < VMCTX_MM_SLOTS; i++)
		if (READ_ONCE(vmctx_mms[i].mm) == mm) {
			vmctx_zeropage_forbidden++;
			return true;
		}
	return false;
}
EXPORT_SYMBOL_GPL(vmctx_mm_forbids_zeropage);

/*
 * One page per fault for a context's memory -- see the declaration.
 *
 * vmctx_batch_install=Y restores the batch installs, so a run with them and a
 * run without them can be measured against each other on one boot.
 */
static bool vmctx_batch_install;
core_param(vmctx_batch_install, vmctx_batch_install, bool, 0644);
static long vmctx_single_installs;
core_param(vmctx_single_installs, vmctx_single_installs, long, 0444);

/*
 * Whether a read-only private file fault asks the monitor (the Firefox fix) or
 * is served from this side's own file as it was before. Default on; the switch
 * exists so the class-aware fix and the pre-change behaviour can be measured
 * against each other on one boot -- vmctx_fault_ro_file=0 is pre-change.
 */
bool vmctx_fault_ro_file = true;
core_param(vmctx_fault_ro_file, vmctx_fault_ro_file, bool, 0644);
EXPORT_SYMBOL_GPL(vmctx_fault_ro_file);

bool vmctx_install_single(struct mm_struct *mm)
{
	if (likely(!atomic_read(&vmctx_nr_active)))
		return false;
	if (vmctx_batch_install)
		return false;
	if (!current->vmctx && !vmctx_mm_is_context(mm))
		return false;
	vmctx_single_installs++;
	return true;
}
EXPORT_SYMBOL_GPL(vmctx_install_single);

/*
 * A page of a context, faulted by somebody who is not that context.
 *
 * The obvious repair -- report it to the context's monitor, exactly as the
 * context's own faults are reported -- deadlocks, and the reason is worth
 * stating because it is what decides the shape of this: the commonest foreign
 * faulter IS the monitor. It is writing the page in order to answer the fault
 * it was just asked about, so asking it about its own write would have it wait
 * for itself. Measured, ten pg3 runs: 21000 to 79000 of these per run.
 *
 * So the answer is not who is faulting but which way:
 *
 *   a WRITE is somebody about to put real bytes at that address -- a monitor's
 *   POKE installing a page it holds. Whatever the kernel allocates here is
 *   covered by the write before anyone reads it, so let it allocate. (A write
 *   of PART of a page still leaves the rest invented; that is the caller's to
 *   avoid, and both monitors already fetch a page before writing a run of it.)
 *
 *   a READ has nothing coming to cover it. What the kernel allocates IS the
 *   answer the reader takes away, and for a context's address space that answer
 *   is a page of zeros invented for memory that lives on another machine --
 *   PRINCIPLES.md §3, in the last place in this design that still did it. There
 *   is nothing to ask (see the deadlock above) so it is refused: the fault
 *   fails, get_user_pages() turns that into a short read, and
 *   access_process_vm() returns what it managed. "This side does not hold that
 *   page" is a sentence every caller of PEEK already knows how to hear, and it
 *   is true.
 *
 * vmctx_foreign_refuse=0 restores the old behaviour, so the two can be measured
 * against each other on one boot.
 */
vm_fault_t vmctx_wp_fault(struct mm_struct *mm, int own, int shared, int filed,
			  int zero)
{
	unsigned long flags;
	int i, theirs = 0;

	if (!atomic_read(&vmctx_nr_active))
		return VM_FAULT_FALLBACK;
	spin_lock_irqsave(&vmctx_mm_lock, flags);
	for (i = 0; i < VMCTX_MM_SLOTS; i++)
		if (vmctx_mms[i].mm == mm) {
			theirs = 1;
			break;
		}
	spin_unlock_irqrestore(&vmctx_mm_lock, flags);
	if (!theirs)
		return VM_FAULT_FALLBACK;
	if (own)
		vmctx_wp_own++;
	else
		vmctx_wp_foreign++;
	if (shared)
		vmctx_wp_shared++;
	if (filed)
		vmctx_wp_filed++;
	if (!zero)
		return VM_FAULT_FALLBACK;
	vmctx_wp_zero++;
	/*
	 * ...but only when somebody other than the context is writing.
	 *
	 * Everything above is argued about foreign faulters, and the refusal
	 * was written as if only they could reach here. They cannot. A context
	 * writing a zero page of its OWN address space is the ordinary
	 * copy-on-write that follows its own read: the read installed the
	 * shared zero page, and whether zeros were the right answer was decided
	 * THEN, by do_anonymous_page and the monitor that declined it. Refusing
	 * the write second-guesses a decision already taken, and there is
	 * nothing left to ask -- the page is present and it is this task's.
	 *
	 * Measured: it costs a browser. netsurf and links2 both die in a VM
	 * context on exactly one refused write --
	 *
	 *   unrecoverable guest #PF at 0x7eee733151a0 err 0x7 (-14)
	 *
	 * err 0x7 being present|write|user, a protection fault, and -14 the
	 * SIGBUS below arriving as EFAULT through fixup_user_fault(). With the
	 * refusal off both browsers render exactly as they do natively (three
	 * windows for netsurf, two for links2, same colours); with it on,
	 * neither draws a pixel. Two arms each, alternating, one boot.
	 *
	 * vmctx_wp_zero_own counts the ones now let through, so "the guard
	 * never fires" stays a measurement rather than an assumption.
	 */
	if (own) {
		vmctx_wp_zero_own++;
		return VM_FAULT_FALLBACK;
	}
	if (!vmctx_wp_zero_refuse)
		return VM_FAULT_FALLBACK;
	vmctx_wp_zero_refused++;
	pr_warn_ratelimited("vmctx: a foreign write to the shared zero page in a context's address space: there is nothing to copy, so this kernel would invent one; refusing instead\n");
	return VM_FAULT_SIGBUS;
}
EXPORT_SYMBOL_GPL(vmctx_wp_fault);

/*
 * The fault path that has no hook at all: a FILE fault on somebody else's
 * address space.
 *
 * vmctx_foreign_anon_fault() below covers do_anonymous_page(), and do_fault()'s
 * own hook is guarded on current->vmctx and on !VM_SHARED. Between them they
 * miss the case that matters most on a destination: every guest page there is a
 * MAP_SHARED mapping of the context's backing memfd (vmctx_back_fault() makes
 * it so), and a monitor reading such a page -- process_vm_readv, PEEK,
 * get_user_pages_remote -- faults the CONTEXT's mm while current is the
 * monitor. Neither hook fires. shmem_fault() then allocates a fresh zero folio
 * and inserts it INTO THE OBJECT, permanently, for an address whose contents
 * may be on the other machine, with nobody asked and nothing recorded.
 *
 * Every sibling context that later maps that offset gets the invented folio,
 * so this is not a private mistake in one task's page table: it replaces the
 * address space's memory.
 *
 * Counted, not refused, and that order was deliberate. A foreign WRITE here is
 * how the monitor installs a page in bulk (process_vm_writev, the chunk
 * install) and MUST succeed; a foreign READ is the dangerous one, but refusing
 * it also breaks page_read_as()'s legitimate fall back to the object. So this
 * said how often it happens first.
 *
 * IT NEVER HAPPENS, and that is the number the switch was waiting for.
 * vmctx_ffile_invent -- foreign reads of a context's file mapping for an offset
 * the object does NOT hold, the only ones that would invent a folio -- reads 0
 * across the suite, the browsers and the page tests, against a live denominator
 * in vmctx_ffile_ctx. So the refusal costs nothing measurable, and the
 * objection above (that it would break page_read_as()'s fall back) is answered:
 * that fall back reads offsets the object HAS, which filemap_get_folio() finds
 * and this hook lets through untouched.
 *
 * REFUSING BY DEFAULT WAS MEASURED FALSE, TWICE, AND IS OFF AGAIN.
 *
 * First attempt, refusing every file mapping: 10 refusals in one suite and
 * tests/rd1 failed both of its file cases, because rd1 maps a REAL FILE whose
 * pages legitimately arrive from disk. Narrowing to shmem_mapping() -- the only
 * kind with nothing behind a hole -- was correct and is kept.
 *
 * Second attempt, refusing shmem holes only: 32 refusals, and rd1 failed 3 of 3
 * rounds, deterministically --
 *
 *   store-forward   FAIL  write(2) from the mapping delivered ||
 *   exit-writeback  FAIL  the file holds || after the process ended
 *
 * -- against 7607 foreign reads let through on real files in the same run. A
 * hole in the backing object is not only an invention about to happen: it is
 * also how a page that a context holds, but that is not resident in the object,
 * is legitimately read back. Refusing at fault time turns that into a short
 * read, the destination can no longer serve the page to the source, and the
 * guest's store is lost -- which is a worse failure than the one being
 * prevented, and a visible one.
 *
 * So the default goes back off and the counters stay. vmctx_ffile_invent says
 * how often the case arises (32 per suite) and it is a real gap, but the fix is
 * for the READER to ask whoever holds the page rather than for the fault to be
 * refused underneath it. vmctx_foreign_file_refuse=1 still turns it on.
 *
 * The path that WAS closed, and stays closed, is the decline: a fault for a page
 * whose bytes exist somewhere now kills the program (VMR_EXIT_INVENTED_PAGE)
 * rather than being answered with zeros.
 */
static long vmctx_ffile_all;	/* foreign file faults, anywhere            */
static long vmctx_ffile_ctx;	/* ...on a live context's address space     */
static long vmctx_ffile_read;	/* ...of those, reads (the invented ones)   */
static long vmctx_ffile_shared;	/* ...of those, on a MAP_SHARED vma         */
static long vmctx_ffile_invent;	/* ...of those reads, the ones that would INVENT */
static long vmctx_ffile_refused;
/* Foreign reads let through because a real filesystem is behind the hole. */
static long vmctx_ffile_filed;
static bool vmctx_foreign_file_refuse;
core_param(vmctx_ffile_all, vmctx_ffile_all, long, 0644);
core_param(vmctx_ffile_ctx, vmctx_ffile_ctx, long, 0644);
core_param(vmctx_ffile_read, vmctx_ffile_read, long, 0644);
core_param(vmctx_ffile_shared, vmctx_ffile_shared, long, 0644);
core_param(vmctx_ffile_invent, vmctx_ffile_invent, long, 0644);
core_param(vmctx_ffile_refused, vmctx_ffile_refused, long, 0644);
core_param(vmctx_ffile_filed, vmctx_ffile_filed, long, 0644);
core_param(vmctx_foreign_file_refuse, vmctx_foreign_file_refuse, bool, 0644);

/* ---- every folio the kernel CREATES in a live context's object ---------- */
/*
 * The hx2 residual (HANDOFF-session35): a context's mapping held an ALL-ZERO
 * folio for a page mid-run, with no userspace install of it in the trail and
 * every instrumented door silent -- MAPOBJ refuses holes (vmctx_mapobj_holes
 * 0), a foreign read of a hole is counted (vmctx_ffile_invent did not move),
 * the decline counters did not move. So a zero folio entered the object by a
 * path nothing counts. A per-door audit is how a door gets overlooked, so
 * this counts at the one place EVERY folio of a shmem object is created --
 * shmem_get_folio_gfp()'s allocation -- and classifies by who asked:
 *
 *   self_user     the context's own user-mode #PF on the host path (a fault
 *                 the module did not take in guest mode)
 *   self_kern     the context task in KERNEL mode: a copy_to_user/put_user
 *                 inside a syscall run locally, a signal frame, the clear-tid
 *                 word, or the module's own fixup_user_fault (lsvc = inside
 *                 vmctx_declined_to_local)
 *   mapper        a task that maps the object but is not a context (vmremote's
 *                 own mapping of it)
 *   foreign_write another task writing through GUP: the chunk install
 *                 (process_vm_writev) -- the legitimate producer
 *   foreign_read  another task READING through GUP: PEEK on a hole
 *   file_write    write(2)/pwrite on the object: the other install
 *   file_falloc   fallocate
 *   file_other    anything else (reads allocate nothing: SGP_READ)
 *
 * Two things make it decisive rather than a census. Every allocation is also
 * stamped into a table keyed by (object, page): the LAST allocation for that
 * page, which is the folio the object holds now, because a folio only ever
 * enters the object through this one function. And TAKEOBJ, when the page
 * it captures is all zero, looks the stamp up and names it: which path made
 * the folio, when, by which task, at what address and instruction -- or says
 * that no allocation is on record (the stamp was overwritten by a collision,
 * or the folio predates the boot's counters). A guest's genuinely-zero page
 * is explained the same way (an install from ABSENT, a file write of zeros);
 * the one the residual is about will name a door nothing counted before.
 *
 * "Live object" = an object some context mm maps at this page (an i_mmap
 * walk bounded to the one index, plus a small cache of objects already seen
 * live, so a file write to an offset nobody has mapped yet is counted too).
 */
enum vmctx_shm_path {
	VMCTX_SHM_SELF_USER,
	VMCTX_SHM_SELF_KERN,
	VMCTX_SHM_MAPPER,
	VMCTX_SHM_FOREIGN_WRITE,
	VMCTX_SHM_FOREIGN_READ,
	VMCTX_SHM_FILE_WRITE,
	VMCTX_SHM_FILE_FALLOC,
	VMCTX_SHM_FILE_OTHER,
	VMCTX_SHM_NPATH,
};
static const char *const vmctx_shm_path_name[VMCTX_SHM_NPATH] = {
	"self_user", "self_kern", "mapper", "foreign_write", "foreign_read",
	"file_write", "file_falloc", "file_other",
};
struct vmctx_shm_stamp {
	struct address_space *mapping;
	pgoff_t index;
	u64 when;			/* jiffies64 */
	unsigned long addr;		/* faulting address (fault paths) */
	unsigned long ip;		/* current's user ip at the time */
	u64 sysnr;			/* orig_ax of current's frame */
	pid_t pid;
	char comm[TASK_COMM_LEN];
	u8 path, order, vmctx, lsvc, msys, write, valid;
};
#define VMCTX_SHM_STAMPS 65536		/* direct-mapped by page index */
static struct vmctx_shm_stamp vmctx_shm_stamps[VMCTX_SHM_STAMPS];
static DEFINE_SPINLOCK(vmctx_shm_lock);
#define VMCTX_SHM_OBJS 8
static struct { struct address_space *mapping; u64 seen; } vmctx_shm_objs[VMCTX_SHM_OBJS];

static long vmctx_shm_live;		/* allocations in a live context's object */
static long vmctx_shm_self_user;
static long vmctx_shm_self_kern;
static long vmctx_shm_self_kern_lsvc;	/* ...of self_kern, inside the declined-to-local service */
static long vmctx_shm_mapper;
static long vmctx_shm_foreign_write;
static long vmctx_shm_foreign_read;
static long vmctx_shm_file_write;
static long vmctx_shm_file_falloc;
static long vmctx_shm_file_other;
static long vmctx_shm_large;		/* order > 0: neighbours invented in one go */
static long vmctx_takeobj_zero;		/* TAKEOBJ captured an all-zero page */
static long vmctx_takeobj_zero_named;	/* ...and the stamp named its maker */
static long vmctx_takeobj_zero_unnamed;	/* ...and no allocation was on record */
core_param(vmctx_shm_live, vmctx_shm_live, long, 0644);
core_param(vmctx_shm_self_user, vmctx_shm_self_user, long, 0644);
core_param(vmctx_shm_self_kern, vmctx_shm_self_kern, long, 0644);
core_param(vmctx_shm_self_kern_lsvc, vmctx_shm_self_kern_lsvc, long, 0644);
core_param(vmctx_shm_mapper, vmctx_shm_mapper, long, 0644);
core_param(vmctx_shm_foreign_write, vmctx_shm_foreign_write, long, 0644);
core_param(vmctx_shm_foreign_read, vmctx_shm_foreign_read, long, 0644);
core_param(vmctx_shm_file_write, vmctx_shm_file_write, long, 0644);
core_param(vmctx_shm_file_falloc, vmctx_shm_file_falloc, long, 0644);
core_param(vmctx_shm_file_other, vmctx_shm_file_other, long, 0644);
core_param(vmctx_shm_large, vmctx_shm_large, long, 0644);
core_param(vmctx_takeobj_zero, vmctx_takeobj_zero, long, 0644);
core_param(vmctx_takeobj_zero_named, vmctx_takeobj_zero_named, long, 0644);
core_param(vmctx_takeobj_zero_unnamed, vmctx_takeobj_zero_unnamed, long, 0644);

static bool vmctx_shm_object_live(struct address_space *mapping, pgoff_t index)
{
	struct vm_area_struct *vma;
	unsigned long flags;
	u64 now = get_jiffies_64();
	bool live = false;
	int i, oldest = 0;

	spin_lock_irqsave(&vmctx_shm_lock, flags);
	for (i = 0; i < VMCTX_SHM_OBJS; i++) {
		if (vmctx_shm_objs[i].mapping == mapping &&
		    time_before64(now, vmctx_shm_objs[i].seen + 60 * HZ)) {
			live = true;
			break;
		}
		if (time_before64(vmctx_shm_objs[i].seen, vmctx_shm_objs[oldest].seen))
			oldest = i;
	}
	spin_unlock_irqrestore(&vmctx_shm_lock, flags);
	if (live)
		return true;

	i_mmap_lock_read(mapping);
	vma_interval_tree_foreach(vma, &mapping->i_mmap, index, index) {
		if (vmctx_mm_is_context(vma->vm_mm)) {
			live = true;
			break;
		}
	}
	i_mmap_unlock_read(mapping);
	if (!live)
		return false;

	spin_lock_irqsave(&vmctx_shm_lock, flags);
	vmctx_shm_objs[oldest].mapping = mapping;
	vmctx_shm_objs[oldest].seen = now;
	spin_unlock_irqrestore(&vmctx_shm_lock, flags);
	return true;
}

/*
 * THE DOOR, named by the census above on kernel #150 (session 36): every
 * all-zero page TAKEOBJ captured of hx2's page was a folio created 0-1 ms
 * earlier by foreign_read -- the monitor's VMCTX_CTL_PEEK (access_process_vm
 * -> GUP read fault -> shmem_fault) of a hole that a TAKEOBJ had punched
 * between the monitor's pagemap check and the read. The allocation is the
 * invention: from then on the object "holds" the page as zeros and every
 * sibling maps them as the address space's memory.
 *
 * vmctx_foreign_file_fault() counted and (with vmctx_foreign_file_refuse)
 * refused this from do_fault(), but it looks the folio up BEFORE the fault
 * proceeds: a folio present at that moment and punched a microsecond later
 * passes it, and shmem_get_folio_gfp()'s own repeat loop then finds the hole
 * and allocates. That is why ffile_invent read 10 while this census counted
 * 18 foreign reads that allocated, and why the old refusal "had nothing to
 * refuse" in session 35. A decision about an allocation has to be made where
 * the allocation is made, after the last lookup -- here.
 *
 * A foreign READ that finds a hole in a context's object answers -EFAULT
 * (VM_FAULT_SIGBUS to GUP, a short count to access_process_vm): the reader is
 * told the page is not here, which is the truth, and asks whoever holds it
 * (vmremote's peek_at counts it as n_peek_short; vmhome's ctx_peek returns the
 * short count). A foreign WRITE is the install and must allocate; the
 * context's own faults and the object's file ops are untouched.
 *
 * The switch this was gated on for the A/B is gone; the measurement is
 * kept below, at the counters.
 */
/*
 * ...and only for the ANONYMOUS objects: the backing memfd and MAP_SHARED
 * anonymous memory, which live on the kernel's internal shm mount
 * (SB_KERNMOUNT) and whose hole has nothing behind it. A named tmpfs file is
 * also shmem, and refusing there was measured wrong on #151 (kernel 36's
 * first form): tests/rd1 maps /tmp/rd1.XXXXXX -- tmpfs on the box -- and the
 * owner's sh_file reclaim PEEKs that file's page before a forwarded write(2)
 * to pull the guest's store over it; told "hole", it pulled nothing and the
 * write delivered zeros (rd1 FAIL, the same shape as the two earlier
 * refusal experiments in vmctx_foreign_file_fault's comment). A file's hole
 * reads as the file's zeros by the file's own semantics, and the monitor's
 * file machinery owns its coherence; the anonymous object IS the memory.
 */
/*
 * The A/B that proved it, and was then deleted (PRINCIPLES: no optional
 * correctness -- the numbers stay, the switch does not). Chains of suite x1
 * + 48x8 hx2/pg3/ws1/sig1 on one box, graded by guest.out:
 *   #150, no refusal (s36a):        34/0  46/48 47/48 48/48 48/48 -- both hx2
 *         fails had a foreign_read-made zero folio of the hx2 page, 1 of 46
 *         passes did;
 *   #151, refusal on ALL shmem (s36b): 33/1 48/48 48/48 48/48 48/48 -- rd1
 *         FAIL (the tmpfs-file case above);
 *   #152, refusal on kern-mounted shmem only (s36c, s36d):
 *                                    34/0  48/48 48/48 48/48 48/48 and
 *                                    34/0  47/48 48/48 48/48 48/48 -- zero
 *         foreign_read allocations in all four pools of both chains, every
 *         all-zero take an install; the one s36d miss an exit-242 NULL
 *         dereference of the pre-existing family (NOTES 6.1a), the run's
 *         only decline being that NULL page, no refusal near it.
 */
static long vmctx_shm_refused;		/* foreign reads of a hole refused here */
static long vmctx_shm_filed;		/* ...let through: a named tmpfs file */
core_param(vmctx_shm_refused, vmctx_shm_refused, long, 0644);
core_param(vmctx_shm_filed, vmctx_shm_filed, long, 0644);

bool vmctx_shmem_may_alloc(struct address_space *mapping, pgoff_t index,
			   struct vm_fault *vmf)
{
	struct vm_area_struct *vma = vmf ? vmf->vma : NULL;

	if (!vma || vma->vm_mm == current->mm || (vmf->flags & FAULT_FLAG_WRITE))
		return true;
	if (!vmctx_mm_is_context(vma->vm_mm))
		return true;
	if (!(mapping->host->i_sb->s_flags & SB_KERNMOUNT)) {
		vmctx_shm_filed++;
		return true;
	}
	vmctx_shm_refused++;
	pr_info_ratelimited("vmctx: SHMEM-REFUSED foreign_read: page 0x%lx of a context's object is a hole; pid %d %s (addr 0x%lx sysnr %ld) is told so instead of a zero folio (#%ld)\n",
			    (unsigned long)index << PAGE_SHIFT,
			    task_pid_nr(current), current->comm,
			    vmf->address,
			    current->mm ? (long)task_pt_regs(current)->orig_ax : -1L,
			    vmctx_shm_refused);
	return false;
}
EXPORT_SYMBOL_GPL(vmctx_shmem_may_alloc);

void vmctx_shmem_alloced(struct address_space *mapping, struct folio *folio,
			 struct vm_fault *vmf, int sgp_write, int sgp_falloc)
{
	struct vm_area_struct *vma = vmf ? vmf->vma : NULL;
	struct vmctx_task *vc = current->vmctx;
	struct pt_regs *regs = current->mm ? task_pt_regs(current) : NULL;
	struct vmctx_shm_stamp *st;
	unsigned long flags;
	pgoff_t index = folio->index;
	int path, order = folio_order(folio);
	bool quiet;

	if (vma) {
		if (vma->vm_mm != current->mm)
			path = (vmf->flags & FAULT_FLAG_WRITE) ?
				VMCTX_SHM_FOREIGN_WRITE : VMCTX_SHM_FOREIGN_READ;
		else if (!vmctx_mm_is_context(vma->vm_mm))
			path = VMCTX_SHM_MAPPER;
		else
			path = (vmf->flags & FAULT_FLAG_USER) ?
				VMCTX_SHM_SELF_USER : VMCTX_SHM_SELF_KERN;
		if (path == VMCTX_SHM_MAPPER) {
			/*
			 * A task that is not a context faulting its OWN mapping
			 * of the object. vmremote maps the own.h ownership
			 * segment, a memfd its contexts inherit, so the i_mmap
			 * walk alone called every first touch of that segment a
			 * context's memory (shm_mapper ~1k per pool on #150).
			 * The guest object is mapped at offset == address in
			 * every context; only that shape is the guest's memory.
			 */
			if (vma->vm_pgoff != (vma->vm_start >> PAGE_SHIFT) ||
			    !vmctx_shm_object_live(mapping, index))
				return;
		} else if (path == VMCTX_SHM_FOREIGN_WRITE ||
			   path == VMCTX_SHM_FOREIGN_READ) {
			if (!vmctx_mm_is_context(vma->vm_mm))
				return;
		}
	} else {
		if (!vmctx_shm_object_live(mapping, index))
			return;
		path = sgp_write ? VMCTX_SHM_FILE_WRITE :
		       sgp_falloc ? VMCTX_SHM_FILE_FALLOC : VMCTX_SHM_FILE_OTHER;
	}

	vmctx_shm_live++;
	switch (path) {
	case VMCTX_SHM_SELF_USER:	vmctx_shm_self_user++; break;
	case VMCTX_SHM_SELF_KERN:
		vmctx_shm_self_kern++;
		if (vc && vc->in_local_service)
			vmctx_shm_self_kern_lsvc++;
		break;
	case VMCTX_SHM_MAPPER:		vmctx_shm_mapper++; break;
	case VMCTX_SHM_FOREIGN_WRITE:	vmctx_shm_foreign_write++; break;
	case VMCTX_SHM_FOREIGN_READ:	vmctx_shm_foreign_read++; break;
	case VMCTX_SHM_FILE_WRITE:	vmctx_shm_file_write++; break;
	case VMCTX_SHM_FILE_FALLOC:	vmctx_shm_file_falloc++; break;
	default:			vmctx_shm_file_other++; break;
	}
	if (order)
		vmctx_shm_large++;

	spin_lock_irqsave(&vmctx_shm_lock, flags);
	st = &vmctx_shm_stamps[hash_long((unsigned long)index ^
					  (unsigned long)mapping,
					  ilog2(VMCTX_SHM_STAMPS))];
	st->mapping = mapping;
	st->index = index;
	st->when = get_jiffies_64();
	st->addr = vma ? vmf->address : 0;
	st->ip = regs ? regs->ip : 0;
	st->sysnr = regs ? (u64)regs->orig_ax : (u64)-1;
	st->pid = task_pid_nr(current);
	memcpy(st->comm, current->comm, TASK_COMM_LEN);
	st->path = path;
	st->order = order;
	st->vmctx = !!vc;
	st->lsvc = vc ? !!vc->in_local_service : 0;
	st->msys = vc ? !!vc->in_monitor_syscall : 0;
	st->write = vma ? !!(vmf->flags & FAULT_FLAG_WRITE) : sgp_write;
	st->valid = 1;
	spin_unlock_irqrestore(&vmctx_shm_lock, flags);

	/* The installs are the noise; everything else is said out loud. */
	quiet = (path == VMCTX_SHM_FOREIGN_WRITE || path == VMCTX_SHM_FILE_WRITE ||
		 path == VMCTX_SHM_FILE_FALLOC) && !order;
	if (!quiet)
		pr_info_ratelimited("vmctx: SHMEM-ALLOC %s: page 0x%lx (order %d) of a live context's object created by pid %d %s, addr 0x%lx ip 0x%lx sysnr %lld vmctx=%d lsvc=%d msys=%d write=%d\n",
				    vmctx_shm_path_name[path],
				    (unsigned long)index << PAGE_SHIFT, order,
				    task_pid_nr(current), current->comm,
				    st->addr, st->ip, (long long)st->sysnr,
				    !!vc, st->lsvc, st->msys, st->write);
}
EXPORT_SYMBOL_GPL(vmctx_shmem_alloced);

/*
 * TAKEOBJ captured an all-zero page: say who made the folio it came from.
 */
static void vmctx_shm_explain_zero(struct address_space *mapping, pgoff_t index,
				   unsigned long addr, struct task_struct *target)
{
	struct vmctx_shm_stamp *st, c;
	unsigned long flags;
	u64 now = get_jiffies_64();

	vmctx_takeobj_zero++;
	spin_lock_irqsave(&vmctx_shm_lock, flags);
	st = &vmctx_shm_stamps[hash_long((unsigned long)index ^
					  (unsigned long)mapping,
					  ilog2(VMCTX_SHM_STAMPS))];
	c = *st;
	spin_unlock_irqrestore(&vmctx_shm_lock, flags);

	if (c.valid && c.mapping == mapping && c.index == index) {
		vmctx_takeobj_zero_named++;
		pr_info_ratelimited("vmctx: ZERO-TAKE 0x%lx pid %d (#%ld): the object's folio was created %llu ms ago by %s -- pid %d %s, addr 0x%lx ip 0x%lx sysnr %lld vmctx=%d lsvc=%d msys=%d write=%d order %d\n",
				    addr, task_pid_nr(target), vmctx_takeobj_zero,
				    (unsigned long long)jiffies64_to_msecs(now - c.when),
				    vmctx_shm_path_name[c.path], c.pid, c.comm,
				    c.addr, c.ip, (long long)c.sysnr, c.vmctx,
				    c.lsvc, c.msys, c.write, c.order);
	} else {
		vmctx_takeobj_zero_unnamed++;
		pr_info_ratelimited("vmctx: ZERO-TAKE 0x%lx pid %d (#%ld): NO allocation of the object's folio on record (stamp %s)\n",
				    addr, task_pid_nr(target), vmctx_takeobj_zero,
				    c.valid ? "overwritten by another page" : "empty");
	}
}

/* ---- the complete accounting: every fault on a context's memory --------- */
/*
 * A per-path audit is how a path gets overlooked, so this counts at the one
 * place EVERY user fault passes through -- handle_mm_fault() -- and the hooks
 * that report to a monitor count themselves. The difference is the number that
 * matters and it cannot be wrong by construction:
 *
 *     faults on a context's memory that NOBODY WAS ASKED ABOUT
 *         = vmctx_f_seen - vmctx_f_reported
 *
 * Broken down so a non-zero difference can be attributed rather than merely
 * noticed: by who faulted (the context itself, or a monitor reaching in), and
 * by what kind of mapping it was (anonymous, private file, shared file). On a
 * destination the third of those is all of guest memory.
 *
 * "Resolved locally" is not automatically wrong -- a context's own backing
 * object IS the answer for its own pages, and asking the monitor about them
 * would be a loop. What the split says is WHERE the unasked ones are, so the
 * ones that should have been asked can be told from the ones that must not be.
 */
/*
 * Clear-tid words written inside the protocol at teardown, so the exit path
 * cannot write them outside it. `failed` counts the ones whose put_user still
 * returned an error -- a genuinely bad pointer, or a fault the monitor could
 * not answer; mm_release() ignored that error too, and so does this.
 */
static long vmctx_cleartid_inband;
static long vmctx_cleartid_failed;
core_param(vmctx_cleartid_inband, vmctx_cleartid_inband, long, 0644);
core_param(vmctx_cleartid_failed, vmctx_cleartid_failed, long, 0644);

/*
 * The drop-and-retry fault path: how often it is taken, how often the caller
 * would not tolerate a dropped lock, and how often the answer was a kill (which
 * the retry pass turns into SIGBUS, with the lock held, where that is legal).
 * vmctx_fault_retry=0 restores the old behaviour for an A/B.
 */
static long vmctx_af_retry;
static long vmctx_af_nowait;
static long vmctx_af_retry_killed;
static long vmctx_af_noretry;
static int  vmctx_fault_retry = 1;
core_param(vmctx_af_retry, vmctx_af_retry, long, 0644);
core_param(vmctx_af_nowait, vmctx_af_nowait, long, 0644);
core_param(vmctx_af_retry_killed, vmctx_af_retry_killed, long, 0644);
core_param(vmctx_af_noretry, vmctx_af_noretry, long, 0644);
core_param(vmctx_fault_retry, vmctx_fault_retry, int, 0644);

static long vmctx_f_seen;	/* faults on a live context's mm, all paths  */
static long vmctx_f_own;	/* ...taken by the context itself            */
static long vmctx_f_foreign;	/* ...taken by somebody else (a monitor)     */
static long vmctx_f_anon;	/* ...on an anonymous vma                    */
static long vmctx_f_filepriv;	/* ...on a private file vma                  */
static long vmctx_f_fileshared;	/* ...on a shared file vma (the guest's own) */
static long vmctx_f_write;	/* ...write faults                           */
static long vmctx_f_exec;	/* ...instruction fetches                    */
static long vmctx_f_reported;	/* ...that a monitor was actually asked about */
core_param(vmctx_f_seen, vmctx_f_seen, long, 0644);
core_param(vmctx_f_own, vmctx_f_own, long, 0644);
core_param(vmctx_f_foreign, vmctx_f_foreign, long, 0644);
core_param(vmctx_f_anon, vmctx_f_anon, long, 0644);
core_param(vmctx_f_filepriv, vmctx_f_filepriv, long, 0644);
core_param(vmctx_f_fileshared, vmctx_f_fileshared, long, 0644);
core_param(vmctx_f_write, vmctx_f_write, long, 0644);
core_param(vmctx_f_exec, vmctx_f_exec, long, 0644);
core_param(vmctx_f_reported, vmctx_f_reported, long, 0644);

void vmctx_fault_seen(struct vm_area_struct *vma, unsigned int flags)
{
	unsigned long irqflags;
	struct mm_struct *mm = vma->vm_mm;
	int i, theirs = 0;

	if (!atomic_read(&vmctx_nr_active))
		return;
	spin_lock_irqsave(&vmctx_mm_lock, irqflags);
	for (i = 0; i < VMCTX_MM_SLOTS; i++)
		if (vmctx_mms[i].mm == mm) {
			theirs = 1;
			break;
		}
	spin_unlock_irqrestore(&vmctx_mm_lock, irqflags);
	if (!theirs)
		return;

	vmctx_f_seen++;
	if (mm == current->mm)
		vmctx_f_own++;
	else
		vmctx_f_foreign++;
	if (!vma->vm_file)
		vmctx_f_anon++;
	else if (vma->vm_flags & VM_SHARED)
		vmctx_f_fileshared++;
	else
		vmctx_f_filepriv++;
	if (flags & FAULT_FLAG_WRITE)
		vmctx_f_write++;
	if (flags & FAULT_FLAG_INSTRUCTION)
		vmctx_f_exec++;
}
EXPORT_SYMBOL_GPL(vmctx_fault_seen);

/* Counted by every hook that actually puts the question to a monitor. */
void vmctx_fault_reported(void)
{
	vmctx_f_reported++;
}
EXPORT_SYMBOL_GPL(vmctx_fault_reported);

vm_fault_t vmctx_foreign_file_fault(struct mm_struct *mm, int write, int shared,
				    struct address_space *mapping, pgoff_t pgoff)
{
	unsigned long flags;
	struct folio *folio;
	int i, theirs = 0;

	if (!atomic_read(&vmctx_nr_active))
		return VM_FAULT_FALLBACK;
	vmctx_ffile_all++;
	spin_lock_irqsave(&vmctx_mm_lock, flags);
	for (i = 0; i < VMCTX_MM_SLOTS; i++)
		if (vmctx_mms[i].mm == mm) {
			theirs = 1;
			vmctx_ffile_ctx++;
			if (!write)
				vmctx_ffile_read++;
			if (shared)
				vmctx_ffile_shared++;
			break;
		}
	spin_unlock_irqrestore(&vmctx_mm_lock, flags);

	if (!theirs || write)
		return VM_FAULT_FALLBACK;

	/*
	 * A foreign read splits in two, and only one half is a problem.
	 *
	 * If the page cache already holds this offset, mapping it is right and
	 * necessary -- that IS the address space's memory and the reader must
	 * see it. If it does not, shmem_fault() is about to ALLOCATE a zero
	 * folio and insert it into the object, permanently, for an address
	 * whose contents may be on the other machine. That is not a read; it is
	 * an invention, and it replaces the address space's memory rather than
	 * one task's page table, because every sibling context maps the same
	 * object at the same offset.
	 *
	 * filemap_get_folio() does not allocate, so asking costs a lookup and
	 * changes nothing. Counted first and refused only when asked to: the
	 * count is the measurement, and the refusal is the A/B against it.
	 */
	if (!mapping)
		return VM_FAULT_FALLBACK;
	/*
	 * ...and only where a hole means INVENTION, which is shmem and nothing
	 * else.
	 *
	 * A real file has a filesystem behind it: an offset missing from the
	 * page cache is read from disk, and refusing that read breaks a
	 * perfectly good access without preventing anything. Only an anonymous
	 * object -- the context's backing memfd, which is what every guest page
	 * on a destination is mapped from -- has nothing behind a hole, so
	 * there the allocation IS the invention.
	 *
	 * Measured, and this is why the test is here: turning the refusal on for
	 * every file mapping refused 10 faults in one suite and failed tests/rd1
	 * on both of its file cases --
	 *
	 *   store-forward   FAIL  write(2) from the mapping delivered ||
	 *   exit-writeback  FAIL  the file holds || after the process ended
	 *
	 * -- because rd1 maps a real file and its pages legitimately arrive from
	 * disk. The claim that the refusal 'costs nothing' rested on
	 * vmctx_ffile_invent reading 0, and that zero had been read mid-suite on
	 * a boot with the refusal off: the check had not run, which is the
	 * oldest way to be sure of something false.
	 */
	if (!shmem_mapping(mapping)) {
		vmctx_ffile_filed++;
		return VM_FAULT_FALLBACK;
	}
	folio = filemap_get_folio(mapping, pgoff);
	if (!IS_ERR_OR_NULL(folio)) {
		folio_put(folio);
		return VM_FAULT_FALLBACK;	/* the object has it: map it */
	}
	vmctx_ffile_invent++;
	if (!vmctx_foreign_file_refuse)
		return VM_FAULT_FALLBACK;
	vmctx_ffile_refused++;
	return VM_FAULT_SIGBUS;
}
EXPORT_SYMBOL_GPL(vmctx_foreign_file_fault);

vm_fault_t vmctx_foreign_anon_fault(struct mm_struct *mm, int write)
{
	unsigned long flags;
	int i, theirs = 0;

	if (!atomic_read(&vmctx_nr_active))
		return VM_FAULT_FALLBACK;
	vmctx_foreign_all++;
	spin_lock_irqsave(&vmctx_mm_lock, flags);
	for (i = 0; i < VMCTX_MM_SLOTS; i++)
		if (vmctx_mms[i].mm == mm) {
			theirs = 1;
			vmctx_foreign_ctx++;
			if (!write)
				vmctx_foreign_read++;
			break;
		}
	spin_unlock_irqrestore(&vmctx_mm_lock, flags);

	if (!theirs || write || !vmctx_foreign_refuse)
		return VM_FAULT_FALLBACK;
	vmctx_foreign_refused++;
	return VM_FAULT_SIGBUS;
}
EXPORT_SYMBOL_GPL(vmctx_foreign_anon_fault);

/*
 * Forget the monitor: drop the task reference taken at ATTACH and wake anything
 * blocked waiting for it. Used on DETACH, on context teardown, and when a
 * monitor stops answering (killed, or wedged) — otherwise every subsequent
 * event would pay the full report timeout.
 */
static void vmctx_drop_monitor(struct vmctx_task *vc)
{
	struct task_struct *mon = vc->monitor;

	if (!mon)
		return;
	vc->monitor = NULL;
	put_task_struct(mon);
}

/* Defined below; vmctx_run_current() needs it to undo a failed start. */
static void vmctx_finish(struct task_struct *t, struct pt_regs *regs);

static void vmctx_task_free(struct kref *kref)
{
	kfree(container_of(kref, struct vmctx_task, kref));
}

static struct vmctx_backend *vmctx_backend;
static DEFINE_MUTEX(vmctx_backend_lock);

/*
 * Number of tasks currently running as VM contexts. Read on every
 * return-to-user to decide whether the switch-path guest hook applies;
 * kept as a plain counter so ordinary tasks pay only a load + branch.
 */
atomic_t vmctx_nr_active = ATOMIC_INIT(0);
EXPORT_SYMBOL_GPL(vmctx_nr_active);

bool vmctx_task_active(struct task_struct *t)
{
	struct vmctx_task *vc = t->vmctx;

	return vc && vc->active;
}
EXPORT_SYMBOL_GPL(vmctx_task_active);

int vmctx_register_backend(struct vmctx_backend *b)
{
	int ret = 0;

	if (!b || !b->create || !b->run || !b->destroy)
		return -EINVAL;
	mutex_lock(&vmctx_backend_lock);
	if (vmctx_backend)
		ret = -EBUSY;
	else
		vmctx_backend = b;
	mutex_unlock(&vmctx_backend_lock);
	pr_info("vmctx: backend %s\n", ret ? "rejected (busy)" : "registered");
	return ret;
}
EXPORT_SYMBOL_GPL(vmctx_register_backend);

void vmctx_unregister_backend(struct vmctx_backend *b)
{
	mutex_lock(&vmctx_backend_lock);
	if (vmctx_backend == b)
		vmctx_backend = NULL;
	mutex_unlock(&vmctx_backend_lock);
	pr_info("vmctx: backend unregistered\n");
}
EXPORT_SYMBOL_GPL(vmctx_unregister_backend);

/*
 * Set up the calling task as a VM context and return. The guest is NOT run
 * here: it runs in the return-to-user / context-switch path
 * (exit_to_user_mode_loop -> vmctx_guest_step). This is what makes guest
 * entry happen at the point the scheduler switches to the task, rather than
 * inside a syscall loop. The syscall appears to block until the guest exits
 * because the return-to-user loop keeps re-entering the guest before it
 * actually lets the task return to user space.
 */
long vmctx_run_current(struct vmctx_run_config __user *ucfg)
{
	struct task_struct *t = current;
	struct vmctx_run_config cfg;
	struct vmctx_backend *b;
	struct vmctx_task *vc;
	long ret;

	if (copy_from_user(&cfg, ucfg, sizeof(cfg)))
		return -EFAULT;
	if (cfg.flags & ~(__u64)(VMCTX_FLAG_USERCODE | VMCTX_FLAG_EXIT_PROCESS |
				 VMCTX_FLAG_REDIRECT_SYSCALL |
				 VMCTX_FLAG_REDIRECT_FAULT |
				 VMCTX_FLAG_WAIT_MONITOR |
				 VMCTX_FLAG_SERVICE |
				 VMCTX_FLAG_RESTORE))
		return -EINVAL;
	if (t->vmctx)
		return -EBUSY;

	/*
	 * A service context never enters a guest, so it needs no backend and
	 * the machine it runs on needs no virtualisation at all. That is the
	 * point of it: the side that owns a program has to hold its memory and
	 * make its syscalls, not execute it.
	 */
	if (cfg.flags & VMCTX_FLAG_SERVICE) {
		b = NULL;
	} else {
		mutex_lock(&vmctx_backend_lock);
		b = vmctx_backend;
		if (b && !try_module_get(b->owner))
			b = NULL;
		mutex_unlock(&vmctx_backend_lock);
		if (!b)
			return -ENODEV;
	}

	vc = kzalloc(sizeof(*vc), GFP_KERNEL);
	if (!vc) {
		module_put(b->owner);
		return -ENOMEM;
	}
	kref_init(&vc->kref);
	vc->backend   = b;
	vc->ucfg      = ucfg;
	vc->max_exits = cfg.max_exits;
	vc->flags     = cfg.flags;
	vc->entry     = cfg.entry;
	vc->backing_fd = cfg.backing_fd;
	vc->shared_fd  = cfg.shared_fd;
	vc->stack     = cfg.stack;
	vc->exit_process = !!(cfg.flags & VMCTX_FLAG_EXIT_PROCESS);
	init_waitqueue_head(&vc->ev_wq);
	init_waitqueue_head(&vc->reply_wq);
	init_waitqueue_head(&vc->sys_wq);
	spin_lock_init(&vc->ev_lock);
	spin_lock_init(&vc->sys_pages_lock);

	t->vmctx = vc;
	if (b) {
		ret = b->create(t);
		if (ret) {
			t->vmctx = NULL;
			kref_put(&vc->kref, vmctx_task_free);
			module_put(b->owner);
			return ret;
		}
	}

	/* Arm the switch-path hook. The guest runs on the way back to user. */
	vc->active = 1;
	vc->noted_mm = t->mm;
	vmctx_mm_note(t->mm);
	atomic_inc(&vmctx_nr_active);

	/*
	 * Optionally hold the guest here until a monitor attaches. The context
	 * exists from this point (so VMCTX_CTL_ATTACH can find it) but has not
	 * executed yet, which closes the race where a short-lived guest runs to
	 * completion before its monitor can attach. If nobody attaches in time,
	 * a context that redirects anywhere is refused (see below) and one that
	 * does not carries on handling its own events.
	 */
	if (cfg.flags & VMCTX_FLAG_WAIT_MONITOR) {
		/*
		 * A restoring monitor has work to do before the guest may run —
		 * it installs the checkpointed registers with SETREGS — so wait
		 * for it to say "go" with an explicit RESUME rather than
		 * starting the moment it attaches. Without this the guest could
		 * begin executing whatever the task's registers happened to hold
		 * and race the restore.
		 */
		long left = wait_event_interruptible_timeout(vc->reply_wq,
			vc->monitor != NULL &&
			(!(vc->flags & VMCTX_FLAG_RESTORE) || vc->released),
			5 * HZ);

		if (!vc->monitor) {
			/*
			 * Nobody came. For a context whose memory and syscalls
			 * live on another machine that is not something to
			 * carry on from: "handling its own events" means this
			 * kernel inventing an anonymous zero page for every
			 * fault and running the program's syscalls -- including
			 * its signal dispositions -- on the wrong machine
			 * entirely. It used to do exactly that, and say so in
			 * one pr_info nobody reads.
			 *
			 * Refuse instead. A context that asked for redirection
			 * has no meaning without the thing it redirects to.
			 */
			if (vc->flags & (VMCTX_FLAG_REDIRECT_FAULT |
					 VMCTX_FLAG_REDIRECT_SYSCALL)) {
				pr_warn("vmctx: pid %d: no monitor attached%s, and this context's memory and syscalls are not this kernel's to supply; refusing to start it\n",
					task_pid_nr(t),
					left == 0 ? " (timeout)" : "");
				vc->exit_status = -ENOENT;
				vmctx_finish(t, NULL);
				return -ENOENT;
			}
			pr_info("vmctx: pid %d: no monitor attached%s; handling its own events\n",
				task_pid_nr(t), left == 0 ? " (timeout)" : "");
		}
	}
	return 0;
}

/*
 * Finish a VM context: write back stats, set the syscall return value (when
 * we have the task's user regs), tear down the guest, and disarm the hook.
 */
static void vmctx_finish(struct task_struct *t, struct pt_regs *regs)
{
	struct vmctx_task *vc = t->vmctx;
	struct vmctx_run_config cfg;

	if (!vc)
		return;

	/*
	 * The last write this context's memory needs, performed HERE -- while
	 * the context is still in the protocol -- instead of by the exit path
	 * fifty lines later, when it is not.
	 *
	 * mm_release() (do_exit -> exit_mm -> exit_mm_release) does
	 * put_user(0, tsk->clear_child_tid) and a FUTEX_WAKE: the
	 * CLONE_CHILD_CLEARTID contract that makes pthread_join() work.
	 * Natively that write cannot be wrong, and the reason is worth stating
	 * because it is exactly what breaks here: current->mm is still
	 * installed at that point, mm_users > 1 says other threads can still
	 * observe it, and put_user is an ordinary user access -- so a page that
	 * is not present simply FAULTS IN through handle_mm_fault() on the one
	 * true address space, and the store lands. It is guaranteed to succeed.
	 *
	 * That guarantee is the hazard. Under vmctx the address may be owned by
	 * the other machine, and by the time mm_release() runs this code has
	 * already unhooked: vmctx_task_exit() is called at do_exit()'s top and
	 * exit_mm() is fifty lines below it, so current->vmctx is NULL and
	 * vmctx_mm_forget() has run. Every guard is keyed on one of those two,
	 * so none fires, and do_anonymous_page() does what it is guaranteed to
	 * do -- it MAKES a page. The store then lands in memory nobody asked
	 * for, both machines hold that address, and they disagree: the source
	 * reads 0 while the guest's copy still holds the old tid. A joiner
	 * reading through the protocol is told nothing has changed.
	 *
	 * So do it while the hooks still work. Here, current->vmctx is set and
	 * the monitor is still attached, so put_user's fault is redirected like
	 * any other, the page is fetched from whichever machine holds it, and
	 * the zero lands in the guest's real page. Then clear the pointer, so
	 * mm_release() finds nothing to do and cannot repeat the write outside
	 * the protocol.
	 *
	 * The mm_users test is kept from mm_release() rather than reasoned
	 * about again: with no other user of the address space there is nobody
	 * left to observe the word, and this is the last thread out.
	 *
	 * The wake is kept, and that is not incidental. Suppressing the exit
	 * path's write WITHOUT doing it here was measured and was worse: the
	 * futex key for anonymous memory is (mm, address), not the physical
	 * page, so the exit path's wake did reach waiters even when its store
	 * went to an invented page. Removing the store alone removed the half
	 * that worked. This moves both halves inside the protocol instead.
	 */
	if (t == current && t->mm && t->clear_child_tid &&
	    vc->monitor && (vc->flags & VMCTX_FLAG_REDIRECT_FAULT) &&
	    atomic_read(&t->mm->mm_users) > 1) {
		u32 __user *ctid = t->clear_child_tid;

		vmctx_cleartid_inband++;
		if (put_user(0, ctid))
			vmctx_cleartid_failed++;
		do_futex(ctid, FUTEX_WAKE, 1, NULL, NULL, 0, 0);
		t->clear_child_tid = NULL;
	}

	if (vc->active) {
		vc->active = 0;
		/*
		 * vc->noted_mm, not t->mm: execve replaces the address space
		 * under a live context, and a decrement that lands on a slot
		 * the matching increment never touched corrupts both.
		 */
		vmctx_mm_forget(vc->noted_mm);
		vc->noted_mm = NULL;
		atomic_dec(&vmctx_nr_active);
	}

	/*
	 * Release any monitor waiting on this context: mark it dead so their
	 * wait predicate fires, then wake them. They hold their own reference,
	 * so the structure stays alive until they are done with it.
	 */
	vmctx_drop_monitor(vc);
	vc->ev_pending = 0;
	vc->dead = 1;
	wake_up_all(&vc->ev_wq);
	wake_up_all(&vc->reply_wq);
	/*
	 * And a monitor waiting for a syscall this context will now never
	 * perform. Its wait tests vc->dead as well as vc->sys_done, but nothing
	 * woke that queue, so an exit_group asked for through VMCTX_CTL_SYSCALL
	 * -- the ordinary way to end a context with the status its program
	 * chose -- left the monitor asleep for the full thirty-second timeout.
	 */
	wake_up_all(&vc->sys_wq);

	if (regs && vc->ucfg) {
		memset(&cfg, 0, sizeof(cfg));
		cfg.max_exits       = vc->max_exits;
		cfg.out_enters      = vc->enters;
		cfg.out_exits       = vc->exits;
		cfg.out_counter     = vc->counter;
		cfg.out_faults      = vc->faults;
		cfg.out_exit_reason = vc->last_exit_reason;
		cfg.out_status      = vc->exit_status;
		if (copy_to_user(vc->ucfg, &cfg, sizeof(cfg))) {
			/* Task is on its way out; nothing more we can do. */
		}
		/* vmctx_run(2) "returns" the guest exit status. */
		regs->ax = (unsigned long)(long)vc->exit_status;
	}

	{
		bool kill_process = vc->exit_process;
		int status = vc->exit_status;

		vmctx_mn_detach(vc);
		if (vc->backend) {
			vc->backend->destroy(t);
			module_put(vc->backend->owner);
		}
		task_lock(t);
		t->vmctx = NULL;
		task_unlock(t);
		kref_put(&vc->kref, vmctx_task_free);

		/*
		 * When the guest *is* the process (it exec'd a program, or the
		 * caller asked for it), a guest exit must end the process: there
		 * is no meaningful user frame left to return to. Everything is
		 * already torn down and t->vmctx cleared, so the do_exit() path
		 * (which calls vmctx_task_exit) has nothing left to do.
		 */
		if (kill_process && regs)
			do_exit((status & 0xff) << 8);
	}
}

/*
 * Run the guest once from the return-to-user loop. Called with interrupts
 * enabled, in the task's own context — so a VMEXIT delivered as a host
 * interrupt (e.g. the timer) is taken normally, and if it set need_resched
 * the enclosing loop's schedule() switches this task out with the guest
 * suspended in its VMCB. On the next switch-in we come back here and
 * re-enter the guest.
 */
/*
 * Perform the syscall a monitor asked for, here, as this task.
 *
 * The registers are borrowed and put back. Syscalls read their arguments out of
 * the frame, and some write it -- so the frame has to be the task's real one
 * rather than a copy, and it has to be restored afterwards or the guest resumes
 * with the arguments of a call it never made.
 *
 * Nothing about which syscall this is appears here: the number goes through the
 * kernel's own table, exactly as a guest's own syscall does.
 */
static void uregs_from_ptregs(struct vmctx_uregs *u, const struct pt_regs *r);

/*
 * Which registers differ between the frame before the call and the frame
 * after. Computed here because both copies are in hand and the comparison is
 * a few loads; the monitor would otherwise need the whole register set twice,
 * across a machine boundary, to learn that a call wrote RAX.
 */
static __u32 vmctx_regs_changed(const struct pt_regs *a, const struct pt_regs *b)
{
	__u32 m = 0;

	if (a->ax  != b->ax)  m |= VMCTX_REG_RAX;
	if (a->bx  != b->bx)  m |= VMCTX_REG_RBX;
	if (a->cx  != b->cx)  m |= VMCTX_REG_RCX;
	if (a->dx  != b->dx)  m |= VMCTX_REG_RDX;
	if (a->si  != b->si)  m |= VMCTX_REG_RSI;
	if (a->di  != b->di)  m |= VMCTX_REG_RDI;
	if (a->bp  != b->bp)  m |= VMCTX_REG_RBP;
	if (a->sp  != b->sp)  m |= VMCTX_REG_RSP;
	if (a->r8  != b->r8)  m |= VMCTX_REG_R8;
	if (a->r9  != b->r9)  m |= VMCTX_REG_R9;
	if (a->r10 != b->r10) m |= VMCTX_REG_R10;
	if (a->r11 != b->r11) m |= VMCTX_REG_R11;
	if (a->r12 != b->r12) m |= VMCTX_REG_R12;
	if (a->r13 != b->r13) m |= VMCTX_REG_R13;
	if (a->r14 != b->r14) m |= VMCTX_REG_R14;
	if (a->r15 != b->r15) m |= VMCTX_REG_R15;
	if (a->ip  != b->ip)  m |= VMCTX_REG_RIP;
	if (a->flags != b->flags) m |= VMCTX_REG_RFLAGS;
	return m;
}

static void vmctx_do_monitor_syscall(struct task_struct *t,
				     struct vmctx_task *vc,
				     struct pt_regs *regs)
{
	struct pt_regs saved, handed;
	unsigned long handed_fs, handed_gs;
	long ret;

	if (!regs)
		return;

	saved = *regs;

	regs->orig_ax = vc->sys_nr;
	regs->ax      = vc->sys_nr;
	regs->di      = vc->sys_args[0];
	regs->si      = vc->sys_args[1];
	regs->dx      = vc->sys_args[2];
	regs->r10     = vc->sys_args[3];
	regs->r8      = vc->sys_args[4];
	regs->r9      = vc->sys_args[5];

	/*
	 * The frame as the call receives it, which is what the answer has to be
	 * compared against.
	 *
	 * Comparing with the frame from before the arguments were placed counts
	 * this function's own stores: setting up write(2) reported RDI, RSI,
	 * RDX, R9, R10 and RAX as "changed by the call", when the call changed
	 * only RAX. tests/../vmsys caught it on the first run.
	 */
	handed = *regs;
	handed_fs = t->thread.fsbase;
	handed_gs = t->thread.gsbase;

	/*
	 * Open the window in which pages this call faults in are held against
	 * hand-over. See the sys_pages comment in the struct: the call is
	 * going to touch what it faulted in, and for O_DIRECT the toucher is a
	 * device that no page table can see coming.
	 */
	spin_lock(&vc->sys_pages_lock);
	vc->sys_npages = 0;
	vc->sys_pages_over = 0;
	vc->in_monitor_syscall = 1;
	spin_unlock(&vc->sys_pages_lock);

	ret = vmctx_dispatch_syscall(regs);

	/*
	 * And then finish the call the way this kernel finishes any syscall.
	 *
	 * A syscall does not end at the handler's return statement. What
	 * follows it on the way back to user mode is the rest of the ABI:
	 * a pending signal is delivered -- a frame pushed onto the program's
	 * own stack and the frame pointed at the handler -- and -ERESTARTSYS
	 * is turned into either a rewind onto the SYSCALL instruction or
	 * -EINTR. Both of those are expressed entirely as changes to this
	 * pt_regs, which is exactly what the monitor is handed back.
	 *
	 * Without this the work still happened, one iteration later, in
	 * __exit_to_user_mode_loop() -- and was then thrown away. The order is
	 * the whole of it: the registers were snapshotted here, the monitor was
	 * woken with them, the loop delivered the signal into the frame
	 * afterwards, and the next forwarded syscall's SETREGS overwrote the
	 * result. Measured on tests/sig1.c: after the guest's raise(3), the
	 * service context stood at rip 0x402070 -- the program's own SIGUSR1
	 * handler -- with a complete signal frame written into the program's
	 * stack at rsp 0x7fffffffe2f8, while the machine running the program
	 * carried on from the instruction after its SYSCALL, never entered the
	 * handler, and died in __printf_buffer with the frame sitting in memory
	 * it believed was free. Three of every five runs; the two that survived
	 * are the ones whose stack the frame happened to miss.
	 *
	 * Nothing here is specific to signals, and nothing decides anything: it
	 * runs the same architecture code the return-to-user path runs, one
	 * step earlier, so that the frame it leaves is the frame the monitor
	 * reports. The owner of the program then does what it does with any
	 * call that moved the program -- hands the register set back -- and the
	 * machine executing the instructions enters the handler knowing nothing
	 * about signals, which is the rule this whole design is built on.
	 *
	 * Drained rather than called once, because delivery can leave a second
	 * signal deliverable and the loop that would have taken it is the loop
	 * whose result is discarded. Bounded so that a flag this kernel cannot
	 * clear cannot wedge a context here.
	 */
	{
		int guard = 0;

		while ((test_thread_flag(TIF_SIGPENDING) ||
			test_thread_flag(TIF_NOTIFY_SIGNAL)) && guard++ < 16)
			arch_do_signal_or_restart(regs);
		/*
		 * The return value is part of that state and moves with it: a
		 * restart puts the syscall number back in AX, and an
		 * interrupted call becomes -EINTR. Reporting the value the
		 * handler returned would contradict the register set reported
		 * beside it, and the guest is given both.
		 */
		ret = (long)regs->ax;
	}

	/*
	 * The call is over -- including the signal delivery above, whose frame
	 * writes may also have faulted pages in -- so nothing will touch what
	 * it faulted in on its behalf any more. Let the pages go; a claim that
	 * was waiting on one of them succeeds on its next ask.
	 */
	/*
	 * Before letting them go: is every page this call WROTE still here?
	 *
	 * The list is thrown away below without ever being looked at, and it is
	 * the only record of what the call touched. A page on it that is no
	 * longer present was taken from this address space while the call was
	 * writing it -- so whatever the call stored into it went to a copy the
	 * guest can no longer see, and the guest reads whatever it had. That is
	 * a LOST WRITE, and it is the shape pg3 fails with under a pool:
	 *
	 *   B round 19815: wanted |B00019815-00019815| got |B00019814-00019814|
	 *
	 * exactly one round stale, with the forwarded read(2) having returned
	 * its full 18 bytes.
	 *
	 * Counted, not fixed, and deliberately: the hold was suspected and
	 * MEASURED INNOCENT first -- 64 pool runs per arm with the TTL at 100ms
	 * and at 10000ms gave 7 failures against 3, which is noise, and one
	 * block came out backwards. So the question "is a written page actually
	 * being taken mid-call?" has to be ASKED rather than assumed, and this
	 * is the cheapest way to ask it: a presence check per written page, once
	 * per forwarded call, on a list that is about to be discarded anyway.
	 *
	 * follow_page-free on purpose: this runs with no mmap_lock held and the
	 * call is over, so it asks the page tables directly and tolerates a
	 * racing answer -- a count that is occasionally one out is still the
	 * difference between zero and not-zero, which is the whole question.
	 */
	if (vc->in_monitor_syscall) {
		struct mm_struct *mm = READ_ONCE(t->mm);
		int i, gone = 0;

		spin_lock(&vc->sys_pages_lock);
		for (i = 0; i < vc->sys_npages; i++)
			if (mm && !vmctx_page_present_in(mm, vc->sys_pages[i]))
				gone++;
		/*
		 * The denominator, and it is not optional. "0 pages were lost"
		 * and "the check never ran" are the same reading without it,
		 * and this project has been fooled by that difference before.
		 */
		vmctx_syscall_pages_checked += vc->sys_npages;
		if (vc->sys_npages)
			vmctx_syscall_calls_checked++;
		if (gone) {
			vmctx_syscall_pages_lost += gone;
			vmctx_syscall_calls_lost++;
			pr_warn_ratelimited("vmctx: pid %d: syscall %llu wrote %d page(s) that were gone by the time it finished (of %d it faulted in to write); anything it stored there went to a copy the guest cannot see\n",
					    task_pid_nr(t), vc->sys_nr, gone,
					    vc->sys_npages);
		}
		spin_unlock(&vc->sys_pages_lock);
	}

	spin_lock(&vc->sys_pages_lock);
	vc->in_monitor_syscall = 0;
	vc->sys_npages = 0;
	vc->sys_pages_over = 0;
	spin_unlock(&vc->sys_pages_lock);

	/*
	 * What the call did to the machine state, before the frame is put
	 * back. The monitor is told and decides; nothing is imposed on a
	 * context by a call it did not make itself.
	 */
	vc->sys_changed = vmctx_regs_changed(&handed, regs);
	uregs_from_ptregs(&vc->sys_regs, regs);
	/*
	 * The bases are not in pt_regs, so they are compared separately or the
	 * mask cannot report the one call whose whole effect is to change one.
	 */
	if (handed_fs != t->thread.fsbase)
		vc->sys_changed |= VMCTX_REG_FSBASE;
	if (handed_gs != t->thread.gsbase)
		vc->sys_changed |= VMCTX_REG_GSBASE;
	vc->sys_regs.fs_base = t->thread.fsbase;
	vc->sys_regs.gs_base = t->thread.gsbase;

	*regs = saved;

	vc->sys_ret     = ret;
	vc->sys_pending = 0;
	vc->sys_done    = 1;
	wake_up(&vc->sys_wq);
}

/*
 * Hold a signal back until this context has somewhere to put it.
 *
 * A signal is delivered at an instruction boundary: the kernel pushes a frame
 * onto the stack the program is standing on, saving the registers it was about
 * to resume with, and points the frame at the handler. Every one of those
 * things is a fact about where the program *is*.
 *
 * A service context is never where the program is. It holds the program's
 * address space, its descriptors and its identity, and another machine executes
 * its instructions; the one moment the two agree is while a forwarded syscall
 * is being performed, because the monitor loads the program's registers into
 * this task before the call and reads them back after it. Everywhere else this
 * task's frame is a leftover — the registers of whatever call it last made —
 * and a signal delivered against it writes a frame at an address the program is
 * not using, saving a resume point the program passed long ago. Nothing
 * complains: the frame is well formed, the delivery is counted, and the signal
 * is simply gone.
 *
 * Measured, on tests/sig1.c's "child" case, where a second process signals a
 * parent that is asleep in usleep(3): hits=0 after 100 waits, in a run whose
 * other six cases all passed. The parent's context took the signal between two
 * forwarded calls, built the frame from the previous call's registers, and the
 * next call's SETREGS overwrote the result.
 *
 * So it waits. vmctx_do_monitor_syscall() runs the same delivery immediately
 * after the call it was held for, on registers that are the program's, and the
 * frame it leaves is reported to the monitor with the rest of the machine
 * state.
 *
 * Only a signal that has somewhere to go is held, and that is the whole of the
 * test below. A frame and a boundary are what *entering a handler* needs; a
 * signal that kills the process, stops it, or is thrown away needs neither, and
 * holding one is not a deferral but a refusal. The first version of this held
 * everything except SIGKILL, and tests/pf3.c said so immediately: the owner
 * answers an unhandled fault by setting the disposition to default and sending
 * the signal, so all three of its children reported "exited 242 rather than
 * dying" -- a program that had earned a SIGSEGV and could not be given one.
 *
 * Task work is not held either. It is not a signal, it is this kernel's own
 * work, and holding it stalls whoever queued it.
 */
bool vmctx_defer_signal_work(struct task_struct *t, unsigned long ti_work)
{
	struct vmctx_task *vc = t->vmctx;
	unsigned long flags;
	sigset_t deliverable;
	bool defer = false;
	int sig;

	if (!vc || !(vc->flags & VMCTX_FLAG_SERVICE))
		return false;
	if (vc->dead || !vc->active)
		return false;
	if (ti_work & _TIF_NOTIFY_SIGNAL)
		return false;
	if (fatal_signal_pending(t))
		return false;

	if (!lock_task_sighand(t, &flags))
		return false;
	sigorsets(&deliverable, &t->pending.signal,
		  &t->signal->shared_pending.signal);
	sigandnsets(&deliverable, &deliverable, &t->blocked);
	for (sig = 1; sig < _NSIG; sig++) {
		__sighandler_t h;

		if (!sigismember(&deliverable, sig))
			continue;
		h = t->sighand->action[sig - 1].sa.sa_handler;
		if (h == SIG_DFL || h == SIG_IGN) {
			/*
			 * Whatever this one does -- kill, stop, or nothing at
			 * all -- it is the kernel's to do now, and doing it
			 * needs no frame. Deliver everything and hold nothing:
			 * a set containing one of these is not a set that can
			 * wait.
			 */
			defer = false;
			break;
		}
		/* A handler to enter, and no boundary to enter it at yet. */
		defer = true;
	}
	unlock_task_sighand(t, &flags);
	/*
	 * A set that turned out to be empty leaves defer false on purpose:
	 * TIF_SIGPENDING is then stale, and only get_signal() clears it.
	 */
	return defer;
}
EXPORT_SYMBOL_GPL(vmctx_defer_signal_work);

void vmctx_guest_step(struct pt_regs *regs)
{
	struct task_struct *t = current;
	struct vmctx_task *vc = t->vmctx;
	int r;

	if (!vc)
		return;
	/*
	 * Before anything else, and even for a context whose guest is not
	 * running: a monitor waiting on this is holding a program's syscall.
	 */
	if (vc->sys_pending)
		vmctx_do_monitor_syscall(t, vc, regs);
	if (!vc->active)
		return;
	/*
	 * A service context parks here and never returns to user mode.
	 *
	 * It holds a program -- its memory, its descriptors, its credentials,
	 * its identity -- for a machine that runs that program's instructions
	 * somewhere else. So the one thing it must never do is execute any of
	 * them itself. Returning from here would do exactly that: the task
	 * would carry on at whatever the program's entry point is.
	 *
	 * So it sleeps until there is a syscall to make, and the loop above
	 * makes it. Interruptibly, because the task must still be killable and
	 * must still take signals like any other.
	 */
	if (vc->flags & VMCTX_FLAG_SERVICE) {
		/*
		 * Killable rather than interruptible, and that is the other
		 * half of vmctx_defer_signal_work().
		 *
		 * A signal that is being held back until the context next
		 * completes a syscall leaves TIF_SIGPENDING set, and an
		 * interruptible wait returns the instant that flag is set. The
		 * loop above would then spin at full speed with the signal
		 * never delivered and nothing else to do. A killable wait
		 * ignores it, which is exactly the deferral this needs — and
		 * still lets a fatal signal through, so the context is as
		 * killable as any other task.
		 */
		wait_event_killable(vc->sys_wq,
				    vc->sys_pending || vc->dead ||
				    !vc->active);
		return;
	}

	/*
	 * execve() replaced the address space the guest was running on. Rebuild
	 * the guest against the new mm, entering at the entry point exec set up
	 * in the task's user frame — so the exec'd program itself runs as the
	 * guest.
	 */
	if (vc->needs_rebuild) {
		vc->needs_rebuild = 0;
		vc->backend->destroy(t);
		vc->entry = regs ? regs->ip : 0;
		vc->stack = regs ? regs->sp : 0;
		vc->exit_process = 1;   /* nothing to return to after exec */
		r = vc->backend->create(t);
		if (r) {
			vc->exit_status = r;
			vmctx_finish(t, regs);
			return;
		}
		pr_debug("vmctx: pid %d rebuilt guest after exec: entry 0x%llx sp 0x%llx\n",
			 task_pid_nr(t), vc->entry, vc->stack);
	}

	vmctx_mn_attach(vc);
	r = vc->backend->run(t);
	if (r == VMCTX_RUN_CONTINUE) {
		if (vc->max_exits && vc->exits >= vc->max_exits) {
			vc->exit_status = 0;
			vmctx_finish(t, regs);
		}
		return;
	}

	/* VMCTX_RUN_EXIT or a negative error ends the guest. */
	if (r < 0)
		vc->exit_status = r;
	vmctx_finish(t, regs);
}

/*
 * fork()/clone(): the child must get its OWN VM context. dup_task_struct()
 * copies task_struct wholesale, so the child arrives here holding the
 * parent's vmctx pointer — it is overwritten unconditionally below, which is
 * also what keeps an ordinary child of a VM context clean.
 */
int vmctx_copy_task(struct task_struct *dst, struct task_struct *src,
		    unsigned long clone_flags)
{
	struct vmctx_task *svc = src->vmctx;
	struct vmctx_task *vc;
	int ret;

	dst->vmctx = NULL;

	if (!svc || !svc->active)
		return 0;
	if (!dst->mm)
		return 0;
	/*
	 * CLONE_VM (a thread) needs nothing special: the child shares the
	 * address space, which is what the guest's CR3 already says, and
	 * copy_thread() has put the new thread's stack in the child's pt_regs.
	 */

	/*
	 * A service context has no backend: nothing is ever entered, so there
	 * is no module holding the entry code and none to pin. Asking for one
	 * here dereferenced NULL, which is a fork of the task that owns a
	 * program -- exactly what a guest fork has to do.
	 */
	if (svc->backend && !try_module_get(svc->backend->owner))
		return 0;

	vc = kzalloc(sizeof(*vc), GFP_KERNEL);
	if (!vc) {
		if (svc->backend)
			module_put(svc->backend->owner);
		return -ENOMEM;
	}

	kref_init(&vc->kref);
	/* Inherit the run parameters, not the counters. */
	vc->backend      = svc->backend;
	vc->flags        = svc->flags;
	vc->backing_fd   = svc->backing_fd;
	/*
	 * Inherited as it is, not copied -- which is the point of it. The
	 * child's own memory is a copy of its parent's; this stays the same
	 * object so a store by one is seen by the other.
	 */
	vc->shared_fd    = svc->shared_fd;
	vc->entry        = svc->entry;
	vc->stack        = svc->stack;
	vc->max_exits    = svc->max_exits;
	vc->exit_process = 1;	/* the child has no vmctx_run frame of its own */
	vc->ucfg         = NULL;	/* do not write stats into the child's mm  */
	init_waitqueue_head(&vc->ev_wq);
	init_waitqueue_head(&vc->reply_wq);
	init_waitqueue_head(&vc->sys_wq);
	spin_lock_init(&vc->ev_lock);
	spin_lock_init(&vc->sys_pages_lock);

	/*
	 * Inherit the monitor. A child born without one handles its own events,
	 * and for a context whose memory or syscalls live somewhere else that
	 * means the kernel answers them locally — a fault becomes a zero page
	 * and the child runs memory nobody supplied. There is no window in
	 * which the monitor could attach instead: the child may fault before
	 * its parent's clone() has even returned.
	 *
	 * Any thread of the monitoring process may drive a context, so one
	 * monitor can service a whole process tree.
	 */
	if (svc->monitor) {
		get_task_struct(svc->monitor);
		vc->monitor = svc->monitor;
	}

	dst->vmctx = vc;
	/*
	 * A backend with no clone is the destination's: it runs guest code but
	 * never builds a child itself, because a clone is the source's to run --
	 * the source's kernel makes the child and the destination stands up the
	 * execution context on instruction. A service context has no backend at
	 * all. Either way there is nothing to clone here.
	 */
	ret = (vc->backend && vc->backend->clone) ?
		vc->backend->clone(dst, src) : 0;
	if (ret) {
		dst->vmctx = NULL;
		kref_put(&vc->kref, vmctx_task_free);
		if (svc->backend)
			module_put(svc->backend->owner);
		return ret;
	}

	vc->active = 1;
	vc->noted_mm = dst->mm;
	vmctx_mm_note(dst->mm);
	atomic_inc(&vmctx_nr_active);
	return 0;
}

/*
 * execve(): called once the new address space is installed. The old guest
 * state describes an mm that no longer exists, so mark the context for
 * rebuild; vmctx_guest_step() re-creates it at the new entry point.
 */
void vmctx_exec_notify(struct task_struct *t)
{
	struct vmctx_task *vc = t->vmctx;

	if (vc && vc->active) {
		vc->needs_rebuild = 1;
		/*
		 * And the address space this context is counted into is not
		 * the one it was counted into: exec built a new mm and freed
		 * the old. Without this the old slot is never given back (64
		 * of them, and the leak is permanent until reboot) and the new
		 * mm is in no slot at all, so every fault in the exec'd
		 * program is invented rather than asked about.
		 */
		if (vc->noted_mm != t->mm) {
			vmctx_mm_forget(vc->noted_mm);
			vc->noted_mm = t->mm;
			vmctx_mm_note(t->mm);
		}
	}
}

/* Called from do_exit() for a task that is killed while running a guest. */
void vmctx_task_exit(struct task_struct *t)
{
	if (t->vmctx)
		vmctx_finish(t, NULL);
}

/* ---- redirection to a monitor process (avm.md objectives 2/3) ---------- */

/*
 * How long to wait between checks that the monitor is still alive — not a
 * deadline for answering. A monitor may take arbitrarily long and still be
 * working correctly: servicing one guest syscall can mean executing a blocking
 * one (pselect, a read on a socket, wait4) on another machine, which finishes
 * when the world says so. Treating slowness as death was fatal — the monitor
 * was dropped mid-page-fault and the guest handed a blank page.
 */
#define VMCTX_MONITOR_POLL (2 * HZ)
/* Say something if one event takes this long, so a real hang stays visible. */
#define VMCTX_MONITOR_LOUD (60 * HZ)
/*
 * A fault is different: no correct resolution of a page fault takes two
 * seconds. The bytes are on this machine or one network hop away, and every
 * step that can stall (a take on a busy page) is itself bounded below a
 * second. So an unanswered fault at two seconds is not slow, it is stuck --
 * and the deadline's job is to turn a silent wedge into a loud, attributed
 * failure: the fault, the monitor, and this task's kernel backtrace, then
 * SIGBUS. The alternative was measured: the wait held mmap_read_lock for an
 * hour and took pkill, ps, rmmod and the whole suite down with it.
 */
/*
 * ...and it is a parameter, because the claim in the paragraph above is exactly
 * the thing in doubt.
 *
 * Note what the loop below actually does with it: the wait is
 * wait_event_*_timeout(..., VMCTX_MONITOR_POLL) and `waited` advances by a
 * whole VMCTX_MONITOR_POLL each time round. Both were 2 * HZ, so the deadline
 * fired on the FIRST poll that came back empty -- a fault answered a hair over
 * two seconds late had no second chance, and the granularity was the whole
 * deadline.
 *
 * Milliseconds, live, and **0 means never**. That last is the arm that matters:
 * with the deadline off, a fault that is merely slow completes and the test
 * passes, while a fault that is genuinely deadlocked hangs until the harness
 * limit instead of dying at two seconds. One boot answers "slow or stuck?"
 * without a rebuild and without arguing from the shape of the failure.
 *
 * vmctx_fault_deadline_hits counts the kills, so an arm that reports no failures
 * can be told from an arm where the deadline simply never fired.
 *
 * 6000, not 2000, because the budgets nest and this is the OUTERMOST: the
 * source's page GET socket (4 s) sits inside it, and the destination's GET
 * handler (pull_wait 500 + claim wait 1000 + take wait 1000 ms) inside that.
 * At 2000 the outer bound was smaller than the middle one, and under pool
 * load a GET that merely ran long ended its context (exit 13, 2 of 256 hx2
 * runs). The tools raise a lower live value to this at start-up
 * (vmctx_deadline_enforce); the default is here so a reboot agrees with them.
 */
static unsigned int vmctx_fault_deadline_ms = 6000;
static long vmctx_fault_deadline_hits;
core_param(vmctx_fault_deadline_ms, vmctx_fault_deadline_ms, uint, 0644);
core_param(vmctx_fault_deadline_hits, vmctx_fault_deadline_hits, long, 0644);
#define VMCTX_FAULT_DEADLINE (msecs_to_jiffies(READ_ONCE(vmctx_fault_deadline_ms)))

/*
 * Publish an event, wake the monitor, and wait for its reply. Returns the
 * monitor's action, or VMCTX_ACT_SELF if there is nobody to ask, the wait was
 * interrupted, or the monitor did not answer in time.
 */
static u32 vmctx_report(struct task_struct *t, struct vmctx_task *vc)
{
	long left;
	u32 action;
	int withdrew = 0;	/* this side took the event back off the queue */

	/* A monitor that is on its way out cannot answer; do not wait for it. */
	if (vc->monitor && ((vc->monitor->flags & PF_EXITING) ||
			    !pid_alive(vc->monitor))) {
		pr_info("vmctx: pid %d: monitor gone; handling its own events\n",
			task_pid_nr(t));
		vmctx_drop_monitor(vc);
		return VMCTX_ACT_SELF;
	}

	vc->ev.pid   = task_pid_nr(t);
	spin_lock(&vc->ev_lock);
	vc->ev_pending    = 1;
	vc->ev_taken      = 0;
	vc->reply_pending = 0;
	spin_unlock(&vc->ev_lock);
	wake_up_interruptible(&vc->ev_wq);

	{
		unsigned long waited = 0;
		int moaned = 0;

		for (;;) {
			/*
			 * Once the monitor has taken the event, this wait must
			 * not be cut short by a signal. Giving up here ends in
			 * VMCTX_ACT_RETRY, which rewinds the instruction and
			 * runs the syscall a second time — but the first one has
			 * already executed on the other machine, with its
			 * effects. A read() that consumed its bytes from a pipe
			 * consumes the next ones on the retry: links reads the
			 * resolver's answer, is interrupted by the SIGCHLD its
			 * own resolver child just sent, reads again, gets the
			 * one trailing byte that followed the answer, and calls
			 * the lookup failed. The signal is not lost by waiting —
			 * it is delivered on the way back to userspace, which is
			 * where a syscall's signals are delivered anyway.
			 *
			 * That makes this sleep TASK_UNINTERRUPTIBLE, which in
			 * turn decides what may wake it: wake_up(), never
			 * wake_up_interruptible(), because that one matches
			 * TASK_INTERRUPTIBLE sleepers and walks past this one.
			 * It did, and the context then slept out the whole poll
			 * interval with its reply already sitting in front of
			 * it — two seconds on a syscall, whenever a signal
			 * arrived while the monitor held the event. A guest
			 * that forks is guaranteed one: SIGCHLD, the moment its
			 * child exits.
			 */
			/*
			 * The uninterruptible form is for syscalls, where a
			 * re-run has effects (the 26-line rationale below).
			 * A FAULT event has no such hazard -- asking for a
			 * page twice is asking for a page -- so a taken fault
			 * still yields to a fatal signal. What that buys is
			 * measured, twice: a monitor that took a fault and
			 * never answered ("monitor still working on one event
			 * after 60s") left this task asleep here holding
			 * mmap_read_lock, and pkill, ps and rmmod queued in
			 * D-state behind it until the box read load 6 at 0%%
			 * CPU. SIGKILL must end that context, and with the
			 * uninterruptible wait it could not.
			 */
			if (vc->ev_taken && vc->ev.type == VMCTX_EV_FAULT)
				left = wait_event_killable_timeout(vc->reply_wq,
						vc->reply_pending || !vc->monitor,
						VMCTX_MONITOR_POLL);
			else if (vc->ev_taken)
				left = wait_event_timeout(vc->reply_wq,
						vc->reply_pending || !vc->monitor,
						VMCTX_MONITOR_POLL);
			else
				left = wait_event_interruptible_timeout(vc->reply_wq,
						vc->reply_pending || !vc->monitor,
						VMCTX_MONITOR_POLL);
			if (vc->reply_pending || !vc->monitor)
				break;
			/*
			 * A taken event is NOT abandoned on a signal, fatal or
			 * not. That branch was here and pg3 convicted it: dmesg
			 * showed "pid 432385: a reply arrived for an event its
			 * waiter had abandoned; the syscall may have run twice"
			 * with no deadline warning before it -- so the context
			 * was killed while its monitor was mid-answer, home's
			 * next syscall for it returned "context gone", every
			 * subsequent TAKE answered -EINVAL (not a VM context),
			 * the pull failed -EFAULT, and the guest died fetching
			 * instructions at 0. The kernel printed the exact
			 * hazard the uninterruptible-wait comment below warns
			 * about.
			 *
			 * Killability is not lost by removing it: the fault
			 * deadline further down ends a genuinely stuck fault at
			 * two seconds regardless of signals, which is what
			 * actually prevents the wedge. The killable sleep stays
			 * so a SIGKILL is not ignored outright; what goes is
			 * treating that signal as licence to abandon work the
			 * monitor has already begun.
			 */
			if (left != 0 && !vc->ev_taken) {
				/*
				 * A signal, and as far as this side can see the
				 * monitor has not started on the event -- so
				 * withdraw it and let the guest re-execute the
				 * instruction once the signal has been handled.
				 *
				 * Withdrawing and the monitor's dequeue settle
				 * the same question, so they are done under one
				 * lock. They used to be separate unlocked reads
				 * with the whole exit path of this wait between
				 * them, and the monitor could take the event in
				 * that gap: the instruction was rewound anyway
				 * and the guest ran a syscall the monitor was at
				 * that moment running. Marking the event
				 * abandoned made the late *reply* be discarded,
				 * which does not unmake the write(2) it had
				 * already performed -- th9 emitted forty lines
				 * and printed forty-one, "thread 36 ok" twice,
				 * about once in fifty runs.
				 *
				 * If it turns out the monitor did claim it, this
				 * is not ours to withdraw: go round again, which
				 * now sleeps uninterruptibly because ev_taken is
				 * set.
				 */
				spin_lock(&vc->ev_lock);
				withdrew = !vc->ev_taken;
				if (withdrew)
					vc->ev_pending = 0;
				spin_unlock(&vc->ev_lock);
				if (withdrew)
					break;
				continue;
			}
			/*
			 * Nothing yet. That only matters if the monitor is gone;
			 * if it is alive, it is still working on this event.
			 */
			if ((vc->monitor->flags & PF_EXITING) ||
			    !pid_alive(vc->monitor))
				break;
			waited += VMCTX_MONITOR_POLL;
			if (vc->ev.type == VMCTX_EV_FAULT &&
			    READ_ONCE(vmctx_fault_deadline_ms) &&
			    waited >= VMCTX_FAULT_DEADLINE) {
				pr_warn("vmctx: pid %d: fault at 0x%llx (err 0x%llx rip 0x%llx) unanswered for %lus by monitor %d (%s); this is a stuck step, not a slow one -- backtrace of the waiter follows, then SIGBUS\n",
					task_pid_nr(t),
					(unsigned long long)vc->ev.fault_addr,
					(unsigned long long)vc->ev.fault_err,
					(unsigned long long)vc->ev.rip,
					waited / HZ,
					vc->monitor ? task_pid_nr(vc->monitor) : -1,
					vc->monitor ? vc->monitor->comm : "?");
				dump_stack();
				/*
				 * ...and the stack of the MONITOR, which is the
				 * half that was missing.
				 *
				 * The waiter's backtrace only ever says "it is
				 * waiting", which was never in doubt. What
				 * decides this is what the monitor is doing
				 * instead of answering: blocked performing some
				 * OTHER context's forwarded syscall, asleep on
				 * a lock, or waiting on the far machine. The
				 * monitor is alive (tested just above), so it
				 * is stuck rather than gone, and its kernel
				 * stack names what it is stuck on.
				 *
				 * ev_taken says whether it ever picked this
				 * event up: "not taken" and "taken and stuck"
				 * are different bugs and the message could not
				 * tell them apart.
				 */
				pr_warn("vmctx: ...the monitor had%s taken this event; its stack:\n",
					vc->ev_taken ? "" : " NOT");
				sched_show_task(vc->monitor);
				vmctx_fault_deadline_hits++;
				spin_lock(&vc->ev_lock);
				vc->ev_abandoned = vc->ev_taken;
				vc->ev_pending = 0;
				spin_unlock(&vc->ev_lock);
				return VMCTX_ACT_KILL;
			}
			if (waited >= VMCTX_MONITOR_LOUD && !moaned) {
				moaned = 1;
				pr_info("vmctx: pid %d: monitor still working on one event after %lus (alive; waiting)\n",
					task_pid_nr(t), waited / HZ);
			}
		}
	}
	if (!vc->reply_pending) {
		/*
		 * Three different things end this wait, and they must not be
		 * treated alike.
		 *
		 * A signal says nothing about the monitor. Deciding otherwise
		 * loses the context's memory: a forked parent is guaranteed a
		 * SIGCHLD when its child exits, and answering its fault in the
		 * kernel means fabricating an anonymous zero page — the real
		 * page is on another machine. It then runs blank memory, dying
		 * somewhere unrelated with nothing in the log. Leave the monitor
		 * alone and let the caller re-take the event once the signal has
		 * been delivered.
		 */
		if (withdrew)
			return VMCTX_ACT_RETRY;
		spin_lock(&vc->ev_lock);
		vc->ev_pending = 0;
		spin_unlock(&vc->ev_lock);
		/* The monitor is gone, or it detached while we waited. */
		pr_warn("vmctx: pid %d: monitor is gone; its events are handled in-kernel from now on\n",
			task_pid_nr(t));
		vc->monitor_lost = 1;
		vmctx_drop_monitor(vc);
		return VMCTX_ACT_SELF;
	}

	action = vc->reply.action;
	vc->reply_pending = 0;
	return action;
}

/*
 * Apply to the guest's address space whatever the reply said happened to it.
 *
 * Runs in the context's own task, which is the only place it can: vm_mmap,
 * vm_munmap and vmctx_mprotect all act on current->mm, and there is no remote
 * form of any of them. The monitor decides *what* changed -- it is the side
 * that performed the call on the machine owning the program -- and says so in
 * three words that name no syscall.
 *
 * SET maps from the backing object at the address that is also its offset, the
 * same identity vmctx_back_fault() uses, so a context that shares the object
 * with its siblings gets the mapping they get. PROT changes permission without
 * touching contents; re-mapping would have been simpler and would have thrown
 * away anything mapped privately.
 */
static void vmctx_apply_map(struct task_struct *t, struct vmctx_task *vc);

/*
 * The mapping change itself, taken by value.
 *
 * It used to read vc->reply directly, which is fine for the context whose reply
 * it is and wrong for anyone else's: a sibling is running its own guest and may
 * have a reply in flight, so borrowing its slot to describe someone else's mmap
 * corrupts it. VMCTX_CTL_APPLYMAP needs to apply a change to a context that did
 * not make the call, so the change travels as an argument.
 */
/*
 * Mapping changes a monitor asked to be applied to a task that is not the
 * caller and that the kernel has no remote form of (SET, PROT): refused, so
 * that vm_mmap()/do_mprotect_pkey() never act on the MONITOR's own mm.
 */
static long vmctx_apply_map_remote_refused;
core_param(vmctx_apply_map_remote_refused, vmctx_apply_map_remote_refused, long, 0644);

static void vmctx_apply_map_rep(struct task_struct *t, struct vmctx_task *vc,
				const struct vmctx_reply *rep)
{
	unsigned long off = rep->map_addr & ~PAGE_MASK;
	unsigned long addr = rep->map_addr & PAGE_MASK;
	unsigned long len = PAGE_ALIGN(rep->map_len + off);
	struct file *bf;
	unsigned long got;
	long r;

	if (rep->map_op == VMCTX_MAP_NONE || !len)
		return;

	switch (rep->map_op) {
	case VMCTX_MAP_UNMAP:
		/*
		 * The context's own mm, whoever is calling. Every other op here
		 * is applied by the context itself at resume, where
		 * current == t and vm_munmap() is right; the ctl form
		 * (VMCTX_CTL_APPLYMAP) runs on the MONITOR's thread, and an
		 * unmap through vm_munmap() there would take ranges out of the
		 * monitor instead of the guest. The monitor needs the cross-
		 * task form for exactly one job: a successful exec replaced
		 * the guest's address space at the source, and every mapping
		 * this context still carries describes the PREVIOUS image --
		 * present stale pages that never fault again and hand the new
		 * image's ld.so the old image's data (firefox's fork+exec
		 * children, byte-identical death at 0x7ffff7fb6188).
		 */
		if (t == current) {
			r = vm_munmap(addr, len);
		} else {
			struct mm_struct *mm = get_task_mm(t);

			if (!mm) {
				r = -ESRCH;
			} else if (mmap_write_lock_killable(mm)) {
				mmput(mm);
				r = -EINTR;
			} else {
				r = do_munmap(mm, addr, len, NULL);
				mmap_write_unlock(mm);
				mmput(mm);
			}
		}
		if (r)
			pr_warn_ratelimited("vmctx: pid %d: cannot unmap 0x%lx+0x%lx for the guest: %ld\n",
					    task_pid_nr(t), addr, len, r);
		return;
	case VMCTX_MAP_ZAP: {
		/*
		 * Drop the PAGES of the range and leave the mappings: the
		 * guest-side application of a source MADV_DONTNEED. The next
		 * touch faults through the ordinary service, which asks the
		 * machine that owns the program and gets the fresh content
		 * (zero-fill for private anonymous, the file's bytes for a
		 * file mapping). A hole in the range is not an error (an
		 * untouched page has nothing to drop).
		 *
		 * The zap loop is written out rather than do_madvise(): the
		 * remote-mm DONTNEED path through do_madvise is one nothing
		 * in-tree takes (io_uring advises its own mm; process_madvise
		 * forbids DONTNEED), and walking it tripped
		 * untagged_addr_remote()'s held-mmap-lock assertion inside
		 * get_untagged_addr() before madvise's own per-VMA locking
		 * had taken anything -- a WARN storm per run on the
		 * destination, and the same assert firing silently tainted
		 * the source. What DONTNEED does to pages IS
		 * zap_page_range_single() per VMA under mmap_read_lock, so
		 * that is what runs, skipping the VMA kinds madvise itself
		 * refuses (pfn/IO, locked, hugetlb).
		 */
		struct mm_struct *mm = t == current ? current->mm :
				       get_task_mm(t);
		struct vm_area_struct *vma;
		unsigned long s, e;

		if (!mm)
			return;
		mmap_read_lock(mm);
		{
			VMA_ITERATOR(vmi, mm, addr);

			for_each_vma_range(vmi, vma, addr + len) {
				if (vma->vm_flags & (VM_PFNMAP | VM_IO |
						     VM_LOCKED | VM_HUGETLB))
					continue;
				s = max(addr, vma->vm_start);
				e = min(addr + len, vma->vm_end);
				if (s < e)
					zap_page_range_single(vma, s, e - s,
							      NULL);
			}
		}
		mmap_read_unlock(mm);
		if (t != current)
			mmput(mm);
		return;
	}
	case VMCTX_MAP_PROT: {
		/*
		 * A page at a time, because this address space has holes in it
		 * by construction.
		 *
		 * The guest's memory here is whatever it has faulted on: a
		 * range of the program as it was loaded exists one page at a
		 * time, as vmctx_back_fault() supplies them. mprotect(2) refuses
		 * a range containing an unmapped page outright -- ENOMEM, and it
		 * changes nothing at all -- so a single call over such a range
		 * applied the protection to none of it.
		 *
		 * That is not a missed optimisation. musl and glibc both make
		 * the relocation-read-only segment read-only during start-up,
		 * and it spans more pages than the program has touched by then:
		 * "cannot set prot 0x1 on 0x4ca000+0x5000 ... -12" is that call,
		 * and after it the segment stayed writable for the rest of the
		 * run.
		 *
		 * A hole is not an error here. It is an address the guest has
		 * not asked for yet, and when it does, the fault path supplies
		 * it -- so the protection that matters is the one on the pages
		 * that exist now.
		 */
		unsigned long p, done = 0, missing = 0;

		/*
		 * Only the task itself can change its own protections:
		 * do_mprotect_pkey() acts on current->mm, and there is no
		 * remote form of it. Applied from VMCTX_CTL_APPLYMAP (the
		 * monitor is current) it would change the MONITOR's mappings
		 * -- or, before the SET guard below existed, the guest mappings
		 * a cross-task SET had put there. The sibling's own next fault
		 * re-establishes the protection the monitor records.
		 */
		if (t != current) {
			vmctx_apply_map_remote_refused++;
			pr_warn_ratelimited("vmctx: pid %d: PROT 0x%x on 0x%lx+0x%lx asked for a task that is not the caller; refused -- the kernel cannot change another task's protections, and must not change the monitor's\n",
					    task_pid_nr(t), rep->map_prot,
					    addr, len);
			return;
		}
		for (p = addr; p < addr + len; p += PAGE_SIZE) {
			r = vmctx_mprotect(p, PAGE_SIZE, rep->map_prot);
			if (!r)
				done++;
			else if (r == -ENOMEM)
				missing++;
			else
				pr_warn_ratelimited("vmctx: pid %d: cannot set prot 0x%x on 0x%lx for the guest: %ld\n",
						    task_pid_nr(t),
						    rep->map_prot, p, r);
		}
		pr_debug("vmctx: pid %d: prot 0x%x on 0x%lx+0x%lx: %lu pages set, %lu not mapped yet\n",
			 task_pid_nr(t), rep->map_prot, addr, len, done, missing);
		return;
	}
	case VMCTX_MAP_SET_SHARED:
	case VMCTX_MAP_SET_POPULATE:
	case VMCTX_MAP_SET: {
		int use = rep->map_op == VMCTX_MAP_SET_SHARED
				? vc->shared_fd : vc->backing_fd;
		unsigned long extra = rep->map_op == VMCTX_MAP_SET_POPULATE
				? MAP_POPULATE : 0;

		/*
		 * The same rule, and this one was measured: vm_mmap() maps
		 * into current->mm. Applied cross-task, a SET for a sibling
		 * context mapped the guest's backing object into the
		 * MONITOR's own address space at the guest address
		 * (photographed on tests/mf2: 7ffff57f0000-7ffff77f4000
		 * rw-s /memfd:vmctx-guest in vmremote itself), and every
		 * context the monitor clone()d afterwards inherited that VMA
		 * -- a forked child among them, which then read its PARENT's
		 * stack through it while its own object held the right bytes.
		 * A sibling not given a SET maps the range on its own next
		 * fault, in its own task, where the kernel can do it.
		 */
		if (t != current) {
			vmctx_apply_map_remote_refused++;
			pr_warn_ratelimited("vmctx: pid %d: SET 0x%lx+0x%lx asked for a task that is not the caller; refused -- vm_mmap would map it into the monitor's own mm\n",
					    task_pid_nr(t), addr, len);
			return;
		}
		/*
		 * A shared mapping with no object is not a mapping to make
		 * quietly private.
		 *
		 * Below, a NULL file means MAP_PRIVATE|MAP_ANONYMOUS, which is
		 * the right answer for a context that was given no memory
		 * object at all. It is the wrong answer for a range the owner
		 * called shared: the guest gets private zeroed memory, every
		 * read of it comes back empty, and nothing anywhere says so.
		 * Refusing leaves no mapping, so the fault path supplies the
		 * page as it would for any address -- wrong too, but visibly.
		 */
		if (rep->map_op == VMCTX_MAP_SET_SHARED && use <= 0) {
			pr_warn_ratelimited("vmctx: pid %d: 0x%lx+0x%lx is shared but this context has no shared object; not mapping it\n",
					    task_pid_nr(t), addr, len);
			return;
		}

		bf = use > 0 ? fget(use) : NULL;
		got = vm_mmap(bf, addr, len, rep->map_prot,
			      (bf ? MAP_SHARED : (MAP_PRIVATE | MAP_ANONYMOUS))
			      | MAP_FIXED, bf ? rep->map_off : 0);
		if (bf)
			fput(bf);
		if (IS_ERR_VALUE(got)) {
			pr_warn_ratelimited("vmctx: pid %d: cannot map 0x%lx+0x%lx prot 0x%x for the guest: %ld\n",
					    task_pid_nr(t), addr, len,
					    rep->map_prot, (long)got);
			return;
		}
		/*
		 * POPULATE, by hand rather than MAP_POPULATE: mm_populate()
		 * refuses to write-fault a shared mapping (dirty accounting),
		 * so it would install READ-ONLY entries and a writable
		 * range's every first store would still exit to the monitor
		 * -- a protection fault per page, which is what this op
		 * exists to remove. The ordinary fault path is run here in
		 * the context's own task with FOLL_WRITE where the mapping
		 * allows writing, exactly as the context's own store would,
		 * and the reference is dropped on the spot: only the page
		 * table entry is the point. The monitor names only pages it
		 * has verified the object holds, so no fault here allocates.
		 */
		if (extra) {
			unsigned int gup = (rep->map_prot & 0x2 /* PROT_WRITE */)
					   ? FOLL_WRITE : 0;
			unsigned long p;

			for (p = addr; p < addr + len; p += PAGE_SIZE) {
				struct page *pg = NULL;

				if (get_user_pages_fast(p, 1, gup, &pg) == 1)
					put_page(pg);
			}
		}
		return;
	}
	default:
		pr_warn_ratelimited("vmctx: pid %d: unknown map op %u in a reply\n",
				    task_pid_nr(t), rep->map_op);
		return;
	}
}

/*
 * There is deliberately no way for this kernel to turn a guest's exception into
 * a signal.
 *
 * vmctx_force_fault() stood here, exported for a backend to call. Nothing
 * called it: an exception is published to the monitor as a raw vector and the
 * monitor answers with a register set. Keeping an exported "make this a signal"
 * primitive invites the destination to decide what a vector means, which is the
 * owner's business and not portable to a destination that is not Linux.
 */

int vmctx_redirect_syscall(struct task_struct *t, struct pt_regs *regs)
{
	struct vmctx_task *vc = t->vmctx;
	u32 action;

	if (!vc || !vc->monitor || !(vc->flags & VMCTX_FLAG_REDIRECT_SYSCALL))
		return 0;

	memset(&vc->ev, 0, sizeof(vc->ev));
	vc->ev.type    = VMCTX_EV_SYSCALL;
	vc->ev.nr      = regs->orig_ax;
	vc->ev.args[0] = regs->di;
	vc->ev.args[1] = regs->si;
	vc->ev.args[2] = regs->dx;
	vc->ev.args[3] = regs->r10;
	vc->ev.args[4] = regs->r8;
	vc->ev.args[5] = regs->r9;
	vc->ev.rip     = regs->ip;
	vc->ev.rsp     = regs->sp;
	vc->ev.rflags  = regs->flags;

	action = vmctx_report(t, vc);
	switch (action) {
	case VMCTX_ACT_DONE:
		regs->ax = vc->reply.retval;
		vmctx_apply_map(t, vc);
		return 1;
	case VMCTX_ACT_KILL:
		return -ECANCELED;
	case VMCTX_ACT_RETRY:
		/*
		 * Nobody serviced this. The module already stepped over the
		 * SYSCALL instruction, so rewind: the guest re-executes it once
		 * the signal has been handled, and the monitor sees it then.
		 * Handling it in the kernel instead would run the guest's
		 * syscall on the wrong machine.
		 */
		regs->ip -= 2;
		return 1;
	default:
		return 0;
	}
}
EXPORT_SYMBOL_GPL(vmctx_redirect_syscall);

/*
 * What this side knows about the page a fault landed on.
 *
 * The monitor otherwise has to infer whether a fault is the protocol's doing
 * or the program's, and guesses when its own records are silent. The two are
 * distinguishable here and nowhere else: the VMA is what the program asked
 * for, the PTE is what coherence has done to it since.
 *
 * Nothing here may block on a lock. This runs while a fault is being reported,
 * and one of the two paths that report faults is the kernel's own -- which
 * arrives holding mmap_read_lock. Taking it again is not slow, it is a
 * deadlock: a writer queued in between makes the second down_read wait for a
 * lock the caller is still holding, and the machine stops. So the mm path says
 * so and is not inspected, and even the other path only tries for the lock.
 */
static void vmctx_page_state(struct vmctx_task *vc, struct mm_struct *mm,
			     unsigned long addr, __u64 *flags, __u64 *vmaprot)
{
	struct vm_area_struct *vma;
	spinlock_t *ptl;
	pte_t *ptep;

	*flags = 0;
	*vmaprot = 0;
	if (!mm || vc->in_mm_fault) {
		*flags = VMCTX_PGS_UNKNOWN;
		return;
	}
	if (!mmap_read_trylock(mm)) {
		*flags = VMCTX_PGS_UNKNOWN;
		return;
	}
	vma = vma_lookup(mm, addr);
	if (vma) {
		*flags |= VMCTX_PGS_MAPPED;
		if (vma->vm_flags & VM_READ)
			*vmaprot |= PROT_READ;
		if (vma->vm_flags & VM_WRITE)
			*vmaprot |= PROT_WRITE;
		if (vma->vm_flags & VM_EXEC)
			*vmaprot |= PROT_EXEC;
		ptep = get_locked_pte(mm, addr, &ptl);
		if (ptep) {
			pte_t e = ptep_get(ptep);

			if (pte_present(e)) {
				*flags |= VMCTX_PGS_PRESENT;
				if (pte_write(e))
					*flags |= VMCTX_PGS_WRITABLE;
			}
			pte_unmap_unlock(ptep, ptl);
		}
	}
	mmap_read_unlock(mm);
}

int vmctx_redirect_fault(struct task_struct *t, struct pt_regs *regs,
			 __u64 vec, __u64 addr, __u64 err)
{
	struct vmctx_task *vc = t->vmctx;
	u32 action;

	if (!vc || !vc->monitor || !(vc->flags & VMCTX_FLAG_REDIRECT_FAULT)) {
		/*
		 * A context that asked for redirected faults keeps its memory
		 * somewhere else, so "handle it here" can only mean inventing a
		 * page. If the monitor was lost rather than detaching on
		 * purpose, end the context and say why instead.
		 */
		if (vc && vc->monitor_lost &&
		    (vc->flags & VMCTX_FLAG_REDIRECT_FAULT)) {
			pr_warn("vmctx: pid %d: fault at 0x%llx with no monitor left; ending the context rather than faking its memory\n",
				task_pid_nr(t), addr);
			return -EIO;
		}
		return 0;
	}

	memset(&vc->ev, 0, sizeof(vc->ev));
	vc->ev.type       = VMCTX_EV_FAULT;
	/*
	 * Which exception this was. The owner decides what a vector
	 * means -- that 14 is a memory fault and becomes one signal
	 * while 13 becomes another is its operating system's rule, not
	 * this one's -- so the number has to reach it unread. Without a
	 * field for it only page faults could be reported at all, and
	 * every other exception ended the context here.
	 */
	vc->ev.nr         = vec;
	vc->ev.fault_addr = addr;
	vc->ev.fault_err  = err;
	/* Whose fault this is, as far as this side can safely look. */
	vmctx_page_state(vc, t->mm, (unsigned long)addr & PAGE_MASK,
			 &vc->ev.args[0], &vc->ev.args[1]);
	/*
	 * What kind of mapping it was taken in, when the mm hook is the
	 * reporter and had the VMA in hand. Zero from every other path; see
	 * VMCTX_PGC_VALID.
	 */
	vc->ev.args[2]    = vc->mm_fault_class;
	vc->ev.rip        = regs->ip;
	vc->ev.rsp        = regs->sp;
	vc->ev.rflags     = regs->flags;

	vmctx_fault_reported();
	action = vmctx_report(t, vc);
	switch (action) {
	case VMCTX_ACT_DONE:
		/*
		 * A fault's answer may change the address space too, as a
		 * syscall's always could. The case that needs it: a range the
		 * owner calls shared has to come from the shared object, and a
		 * context that inherited the mapping from its parent got it
		 * from the copy instead -- so the first touch of it is the
		 * moment to say where it really comes from. Nothing else about
		 * this path knows what shared means.
		 */
		vmctx_apply_map(t, vc);
		return 1;		/* monitor made the memory good */
	case VMCTX_ACT_KILL:
		return -ECANCELED;
	case VMCTX_ACT_RETRY:
		/*
		 * Nobody serviced this fault. Re-take it: returning to the exit
		 * loop delivers the pending signal, and the same instruction
		 * faults again afterwards. The one thing we must not do is
		 * service it here — for a context whose memory lives elsewhere,
		 * that means handing the guest a blank page.
		 */
		return 1;
	default:
		return 0;
	}
}
EXPORT_SYMBOL_GPL(vmctx_redirect_fault);

/*
 * Instrument (kernel #93): name the "fault the monitor never saw" that installs a
 * stale local copy of a page the other machine also holds = the hx2 both-holder.
 * vmctx_anon_fault falls back to the local kernel (serves whatever is here,
 * possibly stale) at four points; a WRITE fault that falls back is the store that
 * can be lost. These count each reason, with a write-fault breakdown of the two
 * that mean the monitor could have been asked but was not / declined. Read the
 * deltas across an hx2 run; the reason whose write-fault count tracks the losses
 * is the producer. core_param, resettable, zero overhead when nobody reads.
 */
static unsigned long vmctx_af_calls;	  /* every anon_fault                */
static unsigned long vmctx_af_fb_nomon;   /* no monitor / no REDIRECT_FAULT  */
static unsigned long vmctx_af_fb_lsvc;	  /* in_local_service (nested fault) */
static unsigned long vmctx_af_fb_foreign; /* fault on an mm that is not ours */
static unsigned long vmctx_af_fb_declined;/* monitor asked, answered SELF(0) */
static unsigned long vmctx_af_wf_lsvc;	  /* ...of those, a WRITE fault      */
static unsigned long vmctx_af_wf_declined;/* ...of those, a WRITE fault      */
core_param(vmctx_af_calls, vmctx_af_calls, ulong, 0644);
core_param(vmctx_af_fb_nomon, vmctx_af_fb_nomon, ulong, 0644);
core_param(vmctx_af_fb_lsvc, vmctx_af_fb_lsvc, ulong, 0644);
core_param(vmctx_af_fb_foreign, vmctx_af_fb_foreign, ulong, 0644);
core_param(vmctx_af_fb_declined, vmctx_af_fb_declined, ulong, 0644);
core_param(vmctx_af_wf_lsvc, vmctx_af_wf_lsvc, ulong, 0644);
core_param(vmctx_af_wf_declined, vmctx_af_wf_declined, ulong, 0644);

enum { AF_NOMON, AF_LSVC, AF_FOREIGN, AF_DECLINED };
static vm_fault_t vmctx_af_fallback(int reason, int write_fault)
{
	switch (reason) {
	case AF_NOMON:    vmctx_af_fb_nomon++;    break;
	case AF_LSVC:     vmctx_af_fb_lsvc++;
			  if (write_fault) vmctx_af_wf_lsvc++;
			  break;
	case AF_FOREIGN:  vmctx_af_fb_foreign++;  break;
	case AF_DECLINED: vmctx_af_fb_declined++;
			  if (write_fault) vmctx_af_wf_declined++;
			  break;
	}
	return VM_FAULT_FALLBACK;
}

/*
 * The page-fault path's way in. See the comment on the declaration: this is how
 * a context's memory is supplied when the fault was not taken in guest mode.
 */
/*
 * The hold a write fault-in takes during an in-flight forwarded syscall, in the
 * one form both fault paths use. See the long note in vmctx_anon_fault(): a page
 * this call is about to WRITE must not change machines mid-call, whether the
 * monitor supplied it or the local kernel did.
 */
static void vmctx_af_sys_hold(struct vmctx_task *vc, unsigned long addr,
			      int ret, int write)
{
	unsigned long pg = addr & PAGE_MASK;
	int i, seen = 0;

	if (!(ret == 1 || ret == 0) || !write ||
	    !READ_ONCE(vc->in_monitor_syscall))
		return;
	spin_lock(&vc->sys_pages_lock);
	if (vc->in_monitor_syscall) {
		for (i = 0; i < vc->sys_npages; i++)
			if (vc->sys_pages[i] == pg) {
				vc->sys_pages_when[i] = ktime_get_ns();
				seen = 1;
				break;
			}
		if (!seen) {
			if (vc->sys_npages < (int)ARRAY_SIZE(vc->sys_pages)) {
				vc->sys_pages_when[vc->sys_npages] =
					ktime_get_ns();
				vc->sys_pages[vc->sys_npages++] = pg;
			} else if (!vc->sys_pages_over) {
				vc->sys_pages_over = 1;
			}
		}
	}
	spin_unlock(&vc->sys_pages_lock);
}

vm_fault_t vmctx_anon_fault(struct vm_fault *vmf)
{
	struct task_struct *t = current;
	struct vmctx_task *vc = t->vmctx;
	int af_wf = !!(vmf->flags & FAULT_FLAG_WRITE);
	int ret;

	vmctx_af_calls++;
	if (!vc || !vc->monitor || !(vc->flags & VMCTX_FLAG_REDIRECT_FAULT))
		return vmctx_af_fallback(AF_NOMON, af_wf);
	/*
	 * The backend is already servicing a fault the monitor declined; it
	 * asked for this page deliberately, so do not ask again.
	 */
	if (vc->in_local_service)
		return vmctx_af_fallback(AF_LSVC, af_wf);
	/*
	 * The monitor is another process and holds no context of its own, so
	 * the write it does to supply the page cannot come back through here.
	 */
	/*
	 * Only this task's own address space is the context's memory. A syscall
	 * may fault on an mm that is not the caller's, and the monitor has no
	 * way to reach one: it supplies pages by writing to the *task*, so a
	 * page it "supplies" for a foreign mm lands somewhere else entirely, the
	 * access is retried, and it faults again forever.
	 *
	 * execve is the case that hangs a real program. It builds the new
	 * program's argv/envp in a fresh bprm->mm before installing it, whose
	 * temporary stack starts at STACK_TOP_MAX - PAGE_SIZE — an address the
	 * old address space very likely also has, so nothing about the fault
	 * looks foreign from outside. A GTK browser meets this the first time it
	 * decodes an image: the loader is spawned through bwrap, that child's
	 * exec never completes, and everything waiting on it stops.
	 *
	 * Letting the local kernel have these faults is not a concession — that
	 * memory *is* this machine's. It was never the context's to serve.
	 */
	if (vmf->vma->vm_mm != t->mm)
		return vmctx_af_fallback(AF_FOREIGN, af_wf);
	/*
	 * An x86 page-fault error code the monitor already understands: bit 1
	 * is "write", bit 0 stays clear because the page is not present.
	 */
	/*
	 * Say where this came from. The caller holds mmap_read_lock and may
	 * hold a page-table lock, so nothing reached from here may take either.
	 */
	/*
	 * The mapping's class travels with the report. This is the one place
	 * the VMA is in hand, and the monitor's answer depends on it: a
	 * private file page never handed over is the file's and the local
	 * kernel serves it (the monitor answers SELF, cheaply); one that WAS
	 * handed over is on the other machine and the file here is stale, so
	 * the monitor pulls it. Without the class it had to pick one policy
	 * for every mapping, and each candidate was measured wrong -- see the
	 * comment in do_fault().
	 */
	vc->mm_fault_class = VMCTX_PGC_VALID |
		(vmf->vma->vm_file ? VMCTX_PGC_FILE : 0) |
		((vmf->vma->vm_flags & VM_SHARED) ? VMCTX_PGC_SHARED : 0) |
		((vmf->vma->vm_flags & VM_WRITE) ? VMCTX_PGC_WRITE : 0);
	/*
	 * Do not hold mmap_lock across the wait for the monitor.
	 *
	 * THE DEADLOCK THIS EXISTS FOR, photographed on ws1 with the fault
	 * deadline turned off so the hang holds still:
	 *
	 *   ctx A (a service context) faults inside a forwarded syscall.
	 *         handle_mm_fault holds mmap_READ_lock and sleeps here.
	 *   vmhome serves exactly that fault with a POKE, which is
	 *         access_process_vm -> __access_remote_vm -> wants mmap_READ_lock.
	 *   ctx B (main) is in mmap() for a new thread stack (0x801000, a
	 *         pthread stack) -> queued for mmap_WRITE_lock.
	 *
	 * rwsem is fair, so the monitor's *new* reader queues BEHIND the pending
	 * writer, the writer waits for the reader that is sleeping here, and that
	 * reader waits for the monitor. Three-way cycle, and nothing is corrupt
	 * -- which is why every content check in this tree reported zero. It
	 * needs only a thread creation to overlap another thread's faulting
	 * forwarded syscall, which is ws1 exactly, and browsers constantly.
	 *
	 * The kernel already has the mechanism: drop the lock and return
	 * VM_FAULT_RETRY, as handle_userfault() does for the same reason. The
	 * access is re-taken afterwards and nothing is held across the wait.
	 *
	 * THE THREE RULES, from userfaultfd's own account of them:
	 *  - only with FAULT_FLAG_ALLOW_RETRY; without it the caller cannot cope
	 *    with a dropped lock, so keep the old blocking behaviour;
	 *  - with FAULT_FLAG_RETRY_NOWAIT, return VM_FAULT_RETRY *without*
	 *    dropping and without sleeping -- nowait means the caller will not
	 *    tolerate either;
	 *  - once the lock is dropped, the answer MUST be VM_FAULT_RETRY.
	 *    Returning VM_FAULT_SIGBUS there would have the caller unlock a lock
	 *    this function already released.
	 *
	 * NO LOST WAKEUP. The event is not published until vmctx_report(), which
	 * is called after the drop, so nothing can answer in the window between
	 * releasing the lock and starting to wait; and once published the wait is
	 * a wait_event on vc->reply_wq, which re-tests vc->reply_pending after
	 * prepare_to_wait_event(). A reply landing between publish and sleep is
	 * seen, not missed.
	 *
	 * NOTHING OF vmf SURVIVES THE DROP. After release_fault_lock() another
	 * thread may munmap the range and free the VMA, so the address and the
	 * write bit are copied out first and vmf is not touched again. A range
	 * that really is unmapped by then makes the retry fault on nothing and
	 * the program earns its SIGSEGV, which is the correct answer and the same
	 * one userfaultfd gives.
	 *
	 * FORWARD PROGRESS. The retry arrives with FAULT_FLAG_TRIED set and takes
	 * the blocking path below, so this can loop at most once. When the
	 * monitor supplied the page the retry finds it present and never reaches
	 * here at all; when the monitor declined, the second ask gets the same
	 * answer and falls through to the local kernel. The cost of the fix is
	 * therefore one extra ask on a declined fault, and none on a served one.
	 */
	if (READ_ONCE(vmctx_fault_retry) &&
	    (vmf->flags & FAULT_FLAG_ALLOW_RETRY) &&
	    !(vmf->flags & FAULT_FLAG_TRIED)) {
		unsigned long fa = vmf->address;
		__u64 fe = (vmf->flags & FAULT_FLAG_WRITE) ? 2 : 0;

		if (vmf->flags & FAULT_FLAG_RETRY_NOWAIT) {
			vc->mm_fault_class = 0;
			vmctx_af_nowait++;
			return VM_FAULT_RETRY;	/* lock kept, per the contract */
		}
		release_fault_lock(vmf);	/* vmf is dead from here */
		/*
		 * ...and the flag that told everything downstream "the caller
		 * holds mm locks, do not take one" is now false, so clear it.
		 * The page-state probe the report carries can take mmap_read
		 * again and give the monitor a real answer instead of UNKNOWN.
		 */
		vc->in_mm_fault = 0;
		vmctx_af_retry++;
		ret = vmctx_redirect_fault(t, task_pt_regs(t), 14, fa, fe);
		vc->mm_fault_class = 0;
		if (ret < 0)
			vmctx_af_retry_killed++;
		vmctx_af_sys_hold(vc, fa, ret, !!fe);
		return VM_FAULT_RETRY;
	}

	vc->in_mm_fault = 1;
	ret = vmctx_redirect_fault(t, task_pt_regs(t), 14, vmf->address,
				   (vmf->flags & FAULT_FLAG_WRITE) ? 2 : 0);
	vc->in_mm_fault = 0;
	vc->mm_fault_class = 0;
	/*
	 * A page supplied *to an in-flight assisted syscall* is recorded, and
	 * held against hand-over until the call ends -- see sys_pages in the
	 * struct. Recorded here, before the faulting access retries, which is
	 * the whole point: get_user_pages() has not pinned it yet, so this
	 * record is the only thing standing between the fault-in and a take
	 * that hands over pre-DMA bytes. Only the address is stored; nothing
	 * here may walk a page table, the caller holds mm locks.
	 *
	 * Write fault-ins only, and that restriction is load-bearing. A page
	 * this call will WRITE -- a read(2) buffer the device is about to DMA
	 * into -- must not change machines mid-call, because the write lands
	 * after the copy leaves. A page this call merely READS imposes
	 * nothing: if it is taken away, the next read faults and asks again.
	 * Kernel #43 held read fault-ins too, and futex_wait(2) said why that
	 * is wrong: it faults in the futex word (a read), blocks, and the hold
	 * then kept the other machine from ever taking the page -- so the
	 * write that would have woken the futex could never be made, th9's
	 * threads never finished, and the teardown that followed left a
	 * context asleep in vmctx_report() holding mmap_read_lock, with
	 * pkill, ps and rmmod queued in D-state behind it. A quiet box, load
	 * six, zero CPU.
	 */
	/*
	 * A write fault-in during an in-flight forwarded syscall is held
	 * against hand-over whether the monitor supplied the page (ret == 1)
	 * or the local kernel did (ret == 0). Either way the page now exists
	 * and the call is about to WRITE it, and either way taking it mid-call
	 * loses that write -- the copy leaves before the store lands, the
	 * dma1 defect. A SELF page is not exempt: "the local kernel can serve
	 * it again" is true of the page's *original* bytes, not of the bytes
	 * the syscall is midway through writing into it.
	 *
	 * (This was briefly reverted to ret == 1 only, on a wrong reading that
	 * it caused the sm2/sm3 regression. It did not -- an unsound file_local
	 * short-cut did, fixed in vmhome -- and the hold is measured to be what
	 * keeps pg1/pg2/th9's shared page coherent under the widened fault
	 * hook, which pulls enough extra pages to expose the window.)
	 */
	vmctx_af_noretry++;
	if ((ret == 1 || ret == 0) && (vmf->flags & FAULT_FLAG_WRITE) &&
	    READ_ONCE(vc->in_monitor_syscall)) {
		unsigned long pg = vmf->address & PAGE_MASK;
		int i, seen = 0;

		spin_lock(&vc->sys_pages_lock);
		if (vc->in_monitor_syscall) {
			for (i = 0; i < vc->sys_npages; i++)
				if (vc->sys_pages[i] == pg) {
					/* a fresh fault renews the window */
					vc->sys_pages_when[i] =
						get_jiffies_64();
					seen = 1;
					break;
				}
			if (!seen) {
				if (vc->sys_npages <
				    (int)ARRAY_SIZE(vc->sys_pages)) {
					vc->sys_pages_when[vc->sys_npages] =
						get_jiffies_64();
					vc->sys_pages[vc->sys_npages++] = pg;
				} else if (!vc->sys_pages_over) {
					vc->sys_pages_over = 1;
					vc->sys_over_when = get_jiffies_64();
					pr_warn_ratelimited("vmctx: pid %d: syscall %llu faulted in more than %zu pages; holding every page for the TTL\n",
							    task_pid_nr(t),
							    vc->sys_nr,
							    ARRAY_SIZE(vc->sys_pages));
				}
			}
		}
		spin_unlock(&vc->sys_pages_lock);
	}
	if (ret == 1)
		return 0;		/* supplied: retry the access */
	/*
	 * The context is ending (a fatal signal while the monitor held the
	 * fault, or the monitor told us to kill it). Falling back would have
	 * do_anonymous_page() invent a zero page as this task's dying act --
	 * the one thing no path here is allowed to do. Fail the fault instead;
	 * the task is already on its way out.
	 */
	if (ret < 0)
		return VM_FAULT_SIGBUS;
	return vmctx_af_fallback(AF_DECLINED, af_wf);
}
EXPORT_SYMBOL_GPL(vmctx_anon_fault);

/* ---- vmctx_ctl(2): the tools a monitor needs -------------------------- */

static void uregs_from_ptregs(struct vmctx_uregs *u, const struct pt_regs *r)
{
	u->rax = r->ax;   u->rbx = r->bx;   u->rcx = r->cx;   u->rdx = r->dx;
	u->rsi = r->si;   u->rdi = r->di;   u->rbp = r->bp;   u->rsp = r->sp;
	u->r8  = r->r8;   u->r9  = r->r9;   u->r10 = r->r10;  u->r11 = r->r11;
	u->r12 = r->r12;  u->r13 = r->r13;  u->r14 = r->r14;  u->r15 = r->r15;
	u->rip = r->ip;   u->rflags = r->flags; u->orig_rax = r->orig_ax;
	u->fs_base = 0;   u->gs_base = 0;	/* filled by the caller below */
}

static void ptregs_from_uregs(struct pt_regs *r, const struct vmctx_uregs *u)
{
	r->ax = u->rax;   r->bx = u->rbx;   r->cx = u->rcx;   r->dx = u->rdx;
	r->si = u->rsi;   r->di = u->rdi;   r->bp = u->rbp;   r->sp = u->rsp;
	r->r8 = u->r8;    r->r9 = u->r9;    r->r10 = u->r10;  r->r11 = u->r11;
	r->r12 = u->r12;  r->r13 = u->r13;  r->r14 = u->r14;  r->r15 = u->r15;
	r->ip = u->rip;   r->flags = u->rflags; r->orig_ax = u->orig_rax;
}

/*
 * Take a page out of the context and hand its contents to the monitor.
 *
 * A page of a context's memory may exist on two machines, and only one of them
 * may hold it writable at a time — otherwise both write and the loser's changes
 * are reconstructed by guessing, which is what a diff against a snapshot is.
 * This is the transfer: the page stops existing here and its last contents go
 * to whoever asked, so there is exactly one copy and no write set to infer.
 *
 * The order is the whole point, and it has three steps rather than two.
 *
 * Reading the page and then dropping it leaves it writable while it is being
 * copied, so another thread of this context can change it in between and have
 * that change discarded. Dropping it first makes the range unreachable —
 * zap_page_range_single() clears the PTEs and flushes the TLB — and the
 * reference taken below is what keeps the page alive afterwards for us to read.
 * No CPU can reach it, so what we read is exactly its final state.
 *
 * All of that is about the copy, and it left the *refusal* unaccounted for: a
 * page may turn out not to be movable, and finding that out after the drop
 * means the page is gone either way. So the drop is now the middle step, and
 * the first one is to ask — vmctx_folio_precheck(), which answers every part of
 * the question that can be answered while the page is still mapped and changes
 * nothing. A refused take leaves the address exactly as it found it; the drop
 * happens only for a take that is going to succeed.
 */
/*
 * Is this page held by an assisted syscall still in flight in this context?
 *
 * If it is, it may not change machines: the call faulted it in to touch it,
 * and the touch may be a device write no page table records. -EBUSY, and the
 * caller waits -- the wait is bounded by the syscall's own duration, because
 * the list empties when the call completes.
 */
/*
 * How long a recorded page stays held. Long enough to cover the measured
 * fault-to-pin window (microseconds; the pin then protects the page by
 * refcount), short enough that a blocking syscall's transient write
 * fault-ins -- futex_wait's own stack, via get_futex_key's write GUP --
 * do not hold the peer's stack hostage against the very wake it sleeps for.
 */
/*
 * How long a syscall's hold on a page lasts, in milliseconds, and it is a
 * parameter so the question "is the TTL the bug?" can be ASKED rather than
 * argued. Default unchanged at 100ms.
 *
 * The hold is renewed only by a FAULT ("a fresh fault renews the window",
 * below), so a call that keeps writing a page already present never renews it
 * and the hold lapses while the call is still writing. pg3 under an 8-way pool
 * shows what that costs: the forwarded read(2) returns its full 18 bytes, and
 * the guest reads the PREVIOUS round's bytes out of its own buffer --
 *   B round 19815: wanted |B00019815-00019815| got |B00019814-00019814|
 * -- a write the source made into a page the guest no longer shared.
 */
static unsigned int vmctx_takeobj_gap_us;	/* widen the named TAKEOBJ window, to test it */
/*
 * mm/shmem.c. No public declaration because nothing outside shmem has needed
 * it; vmctx is built into the kernel, so declaring it here is the same linkage
 * shmem_undo_range() uses.
 */
bool shmem_recalc_inode(struct inode *inode, long alloced, long swapped);

static long vmctx_takeobj_gaps;		/* hand-overs past the point of no return */
static long vmctx_takeobj_raced;	/* ...where someone mapped it in the gap */
/*
 * No longer the primary release -- a SAFETY NET. A forwarded syscall's write
 * hold is now lifted by an event (the call blocks or returns; see
 * vmctx_syscall_holds_one), not by a timer, so this only fires when a page has
 * been held while the call was RUNNING for this long without blocking or
 * returning -- which is a red flag, logged, not a routine expiry. Generous
 * because a legitimately long in-kernel syscall that is actively running should
 * not be cut off early; if this fires, something is wrong upstream.
 */
static unsigned int vmctx_hold_ttl_ms = 2000;
static long vmctx_syscall_hold_ttl_fired;   /* safety TTL lapsed: investigate  */
static long vmctx_syscall_hold_blk_release; /* hold lifted because call blocked */
core_param(vmctx_syscall_hold_ttl_fired, vmctx_syscall_hold_ttl_fired, long, 0644);
core_param(vmctx_syscall_hold_blk_release, vmctx_syscall_hold_blk_release, long, 0644);
/*
 * A/B: whether TAKEOBJ punches the folio out of the object after copying it.
 * Default 1 (punch) is the shipped behaviour -- it forces a faulter to the owner
 * (the source) instead of reading a stale in-object copy, the fix hx1 needed. Set
 * to 0 to leave the folio as a stale fallback: this removes the site8 zero-clobber
 * (an emptied object handed zeros over live slots when the source answered ABSENT)
 * at the RISK of reintroducing that very stale-read, which is exactly what this
 * A/B measures.
 */
static bool vmctx_takeobj_punch = true;
core_param(vmctx_hold_ttl_ms, vmctx_hold_ttl_ms, uint, 0644);
core_param(vmctx_takeobj_gap_us, vmctx_takeobj_gap_us, uint, 0644);
core_param(vmctx_takeobj_gaps, vmctx_takeobj_gaps, long, 0644);
core_param(vmctx_takeobj_raced, vmctx_takeobj_raced, long, 0644);
core_param(vmctx_takeobj_punch, vmctx_takeobj_punch, bool, 0644);

#define VMCTX_HOLD_TTL (msecs_to_jiffies(READ_ONCE(vmctx_hold_ttl_ms)))

/*
 * Is this address backed by a present page in that mm right now?
 *
 * A plain page-table walk, taken with no lock: the caller uses it to COUNT a
 * condition, not to decide anything, and a racing answer changes a number by
 * one rather than changing behaviour. Anything stronger would need the
 * mmap_lock in a path that has just finished a syscall and holds nothing.
 */
static bool vmctx_page_present_in(struct mm_struct *mm, unsigned long addr)
{
	pgd_t *pgd; p4d_t *p4d; pud_t *pud; pmd_t *pmd; pte_t *pte;
	spinlock_t *ptl;
	bool present = false;

	pgd = pgd_offset(mm, addr);
	if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd)))
		return false;
	p4d = p4d_offset(pgd, addr);
	if (p4d_none(*p4d) || unlikely(p4d_bad(*p4d)))
		return false;
	pud = pud_offset(p4d, addr);
	if (pud_none(*pud) || unlikely(pud_bad(*pud)))
		return false;
	pmd = pmd_offset(pud, addr);
	if (pmd_none(*pmd) || unlikely(pmd_bad(*pmd)))
		return false;
	pte = pte_offset_map_lock(mm, pmd, addr, &ptl);
	if (!pte)
		return false;
	present = pte_present(ptep_get(pte));
	pte_unmap_unlock(pte, ptl);
	return present;
}

static int vmctx_syscall_holds_one(struct task_struct *t, struct vmctx_task *vc,
				   unsigned long pg)
{
	int i, registered = 0;
	__u64 when = 0;

	if (!vc || !READ_ONCE(vc->in_monitor_syscall))
		return 0;
	spin_lock(&vc->sys_pages_lock);
	if (vc->in_monitor_syscall) {
		if (vc->sys_pages_over) {
			registered = 1;
			when = vc->sys_over_when;
		}
		for (i = 0; !registered && i < vc->sys_npages; i++)
			if (vc->sys_pages[i] == pg) {
				registered = 1;
				when = vc->sys_pages_when[i];
			}
	}
	spin_unlock(&vc->sys_pages_lock);
	if (!registered)
		return 0;

	/*
	 * Event-bounded, not time-bounded. A page a forwarded syscall WROTE
	 * stays held for exactly as long as the call could write it again: while
	 * its thread is running the call (task_is_running) or parked in vmctx
	 * serving one of the call's OWN faults (ev_pending -- the wait inside
	 * vmctx_report, which is not the call giving up the page but the machine
	 * fetching it). The instant the thread blocks in the guest's own wait --
	 * futex, poll, a read with no data: sleeping, and NOT in a vmctx serve --
	 * it cannot be writing this page, so the hold lifts and the peer may take
	 * it. If the call then resumes and writes the page again it re-faults and
	 * re-claims, so nothing is lost; what is bought is that a page in active
	 * use does not ping-pong mid-call, and a page whose call has blocked does
	 * not pin the peer -- the futex-wait deadlock a read-hold once caused, and
	 * the whole reason a bare timeout was ever reached for.
	 */
	if (!task_is_running(t) && !READ_ONCE(vc->ev_pending)) {
		vmctx_syscall_hold_blk_release++;
		return 0;
	}

	/*
	 * The safety net, and a red flag. The block-detection above lifts the
	 * hold on any real block, so reaching this means the page has been held
	 * while the call was RUNNING (or serving) for the whole TTL without
	 * blocking or returning -- a stuck or pathologically long in-kernel
	 * syscall, or a state this code failed to read as blocked. Say which page
	 * and which call, then release rather than pin the peer forever. A store
	 * that lands after this is a genuine lost write, which is exactly why the
	 * event must be surfaced instead of silently tolerated.
	 */
	if (when && time_after64(get_jiffies_64(), when + VMCTX_HOLD_TTL)) {
		vmctx_syscall_hold_ttl_fired++;
		pr_warn_ratelimited("vmctx: pid %d: syscall %llu has held page 0x%lx for over %ums while %s, without blocking or returning -- releasing on the safety TTL; a store to this page after now is lost. Investigate a stuck or very long in-kernel forwarded syscall.\n",
				    task_pid_nr(t), vc->sys_nr, pg,
				    jiffies_to_msecs(VMCTX_HOLD_TTL),
				    task_is_running(t) ? "running" :
							 "serving a fault");
		return 0;
	}
	return 1;
}

/*
 * The whole thread group, not just the named context. A program's threads are
 * separate contexts sharing one mm, and a take names whichever context the
 * caller had in hand -- the measured case (dma1, two threads): the read(2)
 * faults its buffer in on context A, the take arrives naming context B, and a
 * check that looked only at B's record stole the page 87us after it was
 * installed, before get_user_pages() could pin it. Same mm, same page, wrong
 * ledger.
 */
static int vmctx_syscall_holds(struct task_struct *target, unsigned long addr)
{
	unsigned long pg = addr & PAGE_MASK;
	struct task_struct *t;
	int held = 0;

	rcu_read_lock();
	for_each_thread(target, t) {
		if (vmctx_syscall_holds_one(t, READ_ONCE(t->vmctx), pg)) {
			held = 1;
			break;
		}
	}
	rcu_read_unlock();
	return held;
}

/*
 * Make a large folio into folios this code can answer for.
 *
 * A take moves one 4 KiB page. vmctx_folio_precheck() decides whether it may,
 * by comparing the folio's mappings and references against what this caller's
 * unmap is going to remove -- and for a large folio that comparison has no
 * answer: folio_mapcount() counts the mappings of the whole folio, and how many
 * of them a zap of one page removes depends on how the folio is mapped. So the
 * precheck used to decline to answer and let the claim decide after the zap,
 * where a refusal costs the page. Measured over one suite run: take_large=6,
 * take_lost=6 -- every page destroyed by a refusal was a large folio.
 *
 * Guessing the count is not available (PRINCIPLES.md §7, and a guess that is
 * wrong here hands a page to another machine while this one can still write
 * it). The memory-level fix is to stop the folio being large: split it, so the
 * page this take is about is a folio of its own and every question above has an
 * exact answer.
 *
 * split_huge_page() wants the folio locked and the caller holding exactly one
 * reference -- which is this caller's FOLL_GET -- and it transfers that
 * reference to @page, so the caller's put_page() stays correct. The folio
 * containing @page is left locked whether the split succeeded or not, so it is
 * unlocked unconditionally.
 *
 * Every way this can fail is -EBUSY, "not yet", with the page still mapped and
 * intact: a folio someone else has locked, a folio with a reference the split's
 * own freeze refuses (-EAGAIN), hugetlb, which does not split at all. The
 * caller waits and asks again, which is PRINCIPLES.md §2 and is exactly what
 * the userspace take loop already does. Nothing here invents or destroys.
 */
static long vmctx_split_large(struct page *page, struct folio **foliop,
			      unsigned long addr)
{
	struct folio *folio = *foliop;
	int order = folio_order(folio);
	int err;

	vmctx_take_large++;

	if (!vmctx_take_split_large)
		return 0;	/* the control: leave it to the claim */

	/*
	 * hugetlb is not splittable, and it is not what a guest's memory is
	 * made of. Say so rather than calling into a split that would warn.
	 */
	if (folio_test_hugetlb(folio)) {
		vmctx_take_split_fail++;
		pr_warn_ratelimited("vmctx: TAKE 0x%lx: hugetlb folio order %d cannot be split; not yet\n",
				    addr, order);
		return -EBUSY;
	}

	/*
	 * Not folio_lock(): this runs with the target's mmap_lock held and the
	 * caller has its own retry loop, so waiting here for whoever holds the
	 * folio is a wait taken in the wrong place. Refuse, and be asked again.
	 */
	if (!folio_trylock(folio)) {
		vmctx_take_split_fail++;
		pr_warn_ratelimited("vmctx: TAKE 0x%lx: large folio order %d locked by someone else; not yet\n",
				    addr, order);
		return -EBUSY;
	}

	err = split_huge_page(page);
	folio_unlock(page_folio(page));

	if (err) {
		vmctx_take_split_fail++;
		pr_warn_ratelimited("vmctx: TAKE 0x%lx: split of order-%d folio refused (%d): refs=%d want=%d mapcount=%d anon=%d -- not yet\n",
				    addr, order, err,
				    folio_ref_count(folio),
				    folio_expected_ref_count(folio),
				    folio_mapcount(folio),
				    folio_test_anon(folio));
		return -EBUSY;
	}

	/*
	 * -ENOMEM can leave the folio split, but not all the way down. Ask the
	 * page what it belongs to now rather than assuming, and refuse if it is
	 * still large -- the precheck below still could not answer for it.
	 */
	folio = page_folio(page);
	if (folio_test_large(folio)) {
		vmctx_take_split_fail++;
		pr_warn_ratelimited("vmctx: TAKE 0x%lx: split left order %d, still large; not yet\n",
				    addr, folio_order(folio));
		return -EBUSY;
	}

	vmctx_take_split++;
	*foliop = folio;
	return 0;
}

/*
 * A page has just been unmapped or write-protected in the host page tables and
 * the host TLBs flushed. That flush reaches only the CPUs in the mm's cpumask
 * and only the host ASID; a guest of THIS or ANY OTHER context may be executing
 * on another CPU right now with the old translation cached in its own
 * ASID-tagged TLB, which no host flush touches. (The shared object is mapped by
 * many single-threaded contexts, each its own mm, so "the mm's cpumask" is not
 * the set of CPUs that can still write the folio.) Ask the backend to force
 * those guests out of guest mode: an entry reloads the guest TLB, so once this
 * returns no guest can reach the page except by faulting. Called while the
 * folio is still pinned, before it is copied or freed, so a store made through a
 * stale mapping between the unmap and this call lands in the folio we still
 * hold and is not lost.
 */
static void vmctx_backend_invalidate(struct task_struct *target,
				     unsigned long addr)
{
	struct vmctx_task *vc = target->vmctx;

	if (vc && vc->backend && vc->backend->invalidate)
		vc->backend->invalidate(target, addr);
}

/* ---- the invalidations vmctx does not perform itself ------------------ */
/*
 * vmctx_backend_invalidate() above is called from the four places this code
 * takes a page away on purpose. That is not the whole set of ways a folio
 * stops being where a running guest's TLB says it is: reclaim, migration,
 * compaction, THP collapse and the guest's own munmap/mprotect all clear PTEs
 * without going anywhere near vmctx, and the guest's ASID-tagged TLB survives
 * every host flush they do.
 *
 * mmu_notifier is the kernel's own mechanism for a secondary TLB that shares
 * the CPU's page tables, which is exactly what an NPT/EPT-identity guest
 * running on the task's own CR3 has. arch_invalidate_secondary_tlbs() is the
 * op meant for it -- it fires from flush_tlb_mm_range() for every invalidation
 * in the mm, not only the ones a caller thought to announce.
 *
 * The shootdown itself is the one that already exists: the backend's kick,
 * which forces every CPU currently inside a guest to take a VMEXIT. It is
 * given no task because there is none to give -- the notifier knows an mm, and
 * the kick is global by nature (the shared object is mapped by many contexts,
 * each its own mm, so "this mm's guests" is not the set that can still reach
 * the folio). A backend's invalidate must therefore tolerate a NULL task.
 */
struct vmctx_mn {
	struct mmu_notifier mn;
	struct mm_struct   *mm;
};

static unsigned long vmctx_mn_calls;	/* notifier reached                  */
static unsigned long vmctx_mn_kicks;	/* ...and a shootdown was issued     */
static unsigned long vmctx_mn_unsafe;	/* ...and the context forbade one    */
/*
 * OFF by default, and that is a measurement decision rather than a doubt.
 *
 * Registration happens from the guest's run loop and is gated on this, so one
 * boot can carry both arms: run the case with it 0, write 1 to
 * /sys/module/kernel/parameters/vmctx_mn_enable, run the case again. Comparing
 * two boots would compare two machines' worth of scheduling noise as well, and
 * this failure is a race measured in single-digit percentages.
 *
 * It also means a kernel that boots with this compiled in is inert until asked,
 * which matters on a box with no console.
 *
 * The deliberate kick has its own switch in the module (kick_enable); this one
 * is the notifier's.
 */
static int vmctx_mn_enable;
core_param(vmctx_mn_calls, vmctx_mn_calls, ulong, 0644);
core_param(vmctx_mn_kicks, vmctx_mn_kicks, ulong, 0644);
core_param(vmctx_mn_unsafe, vmctx_mn_unsafe, ulong, 0644);
core_param(vmctx_mn_enable, vmctx_mn_enable, int, 0644);

static void vmctx_mn_arch_invalidate(struct mmu_notifier *sub,
				     struct mm_struct *mm,
				     unsigned long start, unsigned long end)
{
	struct vmctx_backend *be;

	vmctx_mn_calls++;
	if (!READ_ONCE(vmctx_mn_enable))
		return;
	be = READ_ONCE(vmctx_backend);
	if (!be || !be->invalidate)
		return;
	/*
	 * The kick is a broadcast IPI that is WAITED ON, and that may not be
	 * issued with interrupts off or from an atomic section -- two CPUs
	 * doing it at once deadlock. x86 calls this hook at the end of
	 * flush_tlb_mm_range(), after put_cpu(), where both are fine; the
	 * header documents other callers that hold a page-table lock. So the
	 * condition is tested rather than assumed, and the refusals are
	 * counted: a large vmctx_mn_unsafe means this hook is mostly firing
	 * somewhere it cannot act, which is a fact about the kernel and not
	 * about the guest.
	 */
	if (!preemptible() || in_interrupt()) {
		vmctx_mn_unsafe++;
		return;
	}
	vmctx_mn_kicks++;
	be->invalidate(NULL, start);
}

static const struct mmu_notifier_ops vmctx_mn_ops = {
	.arch_invalidate_secondary_tlbs = vmctx_mn_arch_invalidate,
};

/*
 * Attached lazily, from the guest's own run loop.
 *
 * mmu_notifier_register() takes mmap_write_lock and must not be called holding
 * any VM lock, and it wants an mm that cannot go away -- current->mm satisfies
 * both, and vmctx_guest_step() runs with current being the context and nothing
 * of the mm held. Doing it at context creation instead would have to cover the
 * clone path too, where the new task is not current and its mm is being built.
 * A context that never runs a guest -- a service context -- never gets here,
 * which is right: it holds a program's memory but executes none of it.
 */
static void vmctx_mn_attach(struct vmctx_task *vc)
{
	struct vmctx_mn *m;
	struct mm_struct *mm = current->mm;

	if (vc->mn || !mm || !READ_ONCE(vmctx_mn_enable))
		return;
	m = kzalloc(sizeof(*m), GFP_KERNEL);
	if (!m)
		return;
	m->mn.ops = &vmctx_mn_ops;
	m->mm = mm;
	if (mmu_notifier_register(&m->mn, mm)) {
		kfree(m);
		return;
	}
	vc->mn = m;
}

static void vmctx_mn_detach(struct vmctx_task *vc)
{
	struct vmctx_mn *m = vc->mn;

	if (!m)
		return;
	vc->mn = NULL;
	/*
	 * m->mm rather than current->mm: this runs on the way out and the task
	 * may already have dropped its mm. The registration took an mmgrab, so
	 * the pointer is valid until the unregister's matching mmdrop, and
	 * unregister copes with a subscription exit_mmap has already released.
	 */
	mmu_notifier_unregister(&m->mn, m->mm);
	kfree(m);
}

/*
 * VMCTX_CTL_MAPOBJ: map a page the object ALREADY HOLDS into the target
 * context, so the page table entry this installs points at the SAME folio every
 * sibling context maps. No bytes are read, none are written, and NOTHING IS
 * ALLOCATED: a hole in the object is answered as 0 bytes, never faulted in.
 */
static void vmctx_apply_map(struct task_struct *t, struct vmctx_task *vc)
{
	vmctx_apply_map_rep(t, vc, &vc->reply);
}

/*
 * How often MAPOBJ was asked to map a hole and refused. Before this counter the
 * same call ALLOCATED: get_user_pages_remote() without FOLL_NOFAULT runs the
 * ordinary fault, and shmem_fault() on a hole does not fail -- it inserts a
 * fresh zero folio into the object, permanently, for an offset whose bytes live
 * on the other machine. Every sibling context maps the same object, so the
 * invention replaced the ADDRESS SPACE's memory, with no byte written by
 * userspace and every install-side detector silent. ARCHITECTURE #9.2g closed
 * the window for one caller with ctl_lock; the write-grant caller
 * (page_make_writable) measured WORSE under that lock and stayed open, and its
 * race is this exact allocation: hx2's "1 zero; stored 9104894 read 0", ws1's
 * __nptl_nthreads reading 0 in every context at once, a thread descriptor's
 * list pointers reading 0 in __nptl_deallocate_stack.
 */
static long vmctx_mapobj_calls;
static long vmctx_mapobj_holes;
core_param(vmctx_mapobj_calls, vmctx_mapobj_calls, long, 0644);
core_param(vmctx_mapobj_holes, vmctx_mapobj_holes, long, 0644);

static long vmctx_ctl_mapobj(struct task_struct *target, void __user *arg)
{
	struct vm_area_struct *vma;
	struct address_space *mapping;
	struct mm_struct *mm;
	struct vmctx_mem m;
	struct folio *folio;
	struct page *page;
	unsigned long addr;
	spinlock_t *ptl;
	pgoff_t index;
	pte_t *ptep, old;
	long ret;

	if (copy_from_user(&m, arg, sizeof(m)))
		return -EFAULT;
	if (m.len < PAGE_SIZE)
		return -EINVAL;
	addr = m.addr & PAGE_MASK;
	vmctx_mapobj_calls++;

	mm = get_task_mm(target);
	if (!mm)
		return -ESRCH;
	mmap_read_lock(mm);

	vma = vma_lookup(mm, addr);
	if (!vma || !vma->vm_file || !(vma->vm_flags & VM_SHARED)) {
		ret = -ENOENT;		/* not object-backed: the caller falls back */
		goto out_mm;
	}
	if (!(vma->vm_flags & VM_WRITE)) {
		ret = -EFAULT;		/* what FOLL_WRITE answered here before */
		goto out_mm;
	}
	mapping = vma->vm_file->f_mapping;
	if (!shmem_mapping(mapping)) {
		ret = -ENOENT;
		goto out_mm;
	}
	index = linear_page_index(vma, addr);

	/*
	 * The whole repair is this lookup: map a folio the object ALREADY HOLDS,
	 * and answer "not there" for a hole. filemap_lock_folio() allocates
	 * nothing, and holding the folio lock across the install is what makes
	 * the test and the map one operation -- a concurrent TAKEOBJ holds this
	 * same lock across its unmap-copy-remove, so this either maps the folio
	 * before the hand-over (whose unmap_mapping_range then removes this PTE
	 * with the rest) or finds the hole after it and says so. There is no
	 * moment in between where an allocation can be conjured, which is the
	 * TOCTOU the callers used to have to close from outside with ctl_lock --
	 * and could not close for the write-grant path at all.
	 */
	folio = filemap_lock_folio(mapping, index);
	if (IS_ERR(folio)) {
		vmctx_mapobj_holes++;
		ret = 0;		/* a hole is an answer, never an allocation */
		goto out_mm;
	}
	if (folio_test_large(folio)) {
		ret = -ENOENT;		/* not answerable per-page; fall back */
		goto out_folio;
	}
	page = folio_page(folio, 0);

	ptep = get_locked_pte(mm, addr, &ptl);
	if (!ptep) {
		ret = -ENOMEM;
		goto out_folio;
	}
	old = ptep_get(ptep);
	if (pte_present(old)) {
		if (pte_pfn(old) != page_to_pfn(page)) {
			/* mapping some other page: not this call's to replace */
			pte_unmap_unlock(ptep, ptl);
			ret = -EBUSY;
			goto out_folio;
		}
		/*
		 * Present and the object's own folio: re-grant the write, which
		 * is the page_make_writable() half of this call's job. The
		 * coherence write-protect is a PTE fact, so making the PTE
		 * writable again is exactly "the protection is spent".
		 */
		ptep_set_access_flags(vma, addr, ptep,
				      pte_mkwrite(pte_mkdirty(pte_mkyoung(old)),
						  vma), 1);
		pte_unmap_unlock(ptep, ptl);
		ret = PAGE_SIZE;
		goto out_folio;
	}
	if (!pte_none(old)) {
		pte_unmap_unlock(ptep, ptl);
		ret = -EBUSY;		/* swap/PROT_NONE state: not ours */
		goto out_folio;
	}

	/*
	 * The install, exactly as set_pte_range() does it for a file fault:
	 * rmap first, then the PTE. The reference filemap_lock_folio() took
	 * becomes the mapping's -- a zap drops it with the rmap -- so this path
	 * deliberately skips the folio_put the others make.
	 */
	{
		pte_t entry = mk_pte(page, vma->vm_page_prot);

		entry = pte_sw_mkyoung(entry);
		entry = pte_mkwrite(pte_mkdirty(entry), vma);
		inc_mm_counter(mm, mm_counter_file(folio));
		folio_add_file_rmap_pte(folio, page, vma);
		set_pte_at(mm, addr, ptep, entry);
		update_mmu_cache(vma, addr, ptep);
	}
	pte_unmap_unlock(ptep, ptl);
	folio_unlock(folio);
	mmap_read_unlock(mm);
	mmput(mm);
	return PAGE_SIZE;

out_folio:
	folio_unlock(folio);
	folio_put(folio);
out_mm:
	mmap_read_unlock(mm);
	mmput(mm);
	return ret;
}

/*
 * VMCTX_CTL_APPLYMAP: apply a mapping change to a context that did not make the
 * call.
 *
 * A guest's threads share one address space. On the source that is one mm, so a
 * thread's mmap/munmap/mprotect is the whole program's by construction. Here it
 * is not: each context is its own task with its own mm, sharing only the backing
 * object, and vmctx_apply_map() runs on the task whose syscall it answered. So a
 * thread that reshaped the address space left every sibling mapping what no
 * longer exists -- measured on netsurf, where a new thread's arena setup (mmap
 * 128MB, two munmaps to trim it, one mprotect) was followed immediately by the
 * MAIN thread dying on a page fault its owner could not answer (exit 242).
 *
 * The monitor knows the members of an address space, so it fans the change out
 * to them with this. Same code path as the reply's map_op, which is the point:
 * the sibling ends up with exactly the mapping the caller got.
 *
 * The change is passed BY VALUE and never through the target's own reply slot.
 * The target is not stopped -- it is a sibling running its own guest, and may
 * have a reply in flight -- so borrowing that slot corrupts it. Doing exactly
 * that is what broke pf1 and pf2 the first time this was tried.
 */
static long vmctx_ctl_applymap(struct task_struct *target, void __user *arg)
{
	struct vmctx_task *vc = target->vmctx;
	struct vmctx_reply m;

	if (!vc)
		return -ESRCH;
	if (copy_from_user(&m, arg, sizeof(m)))
		return -EFAULT;
	if (m.map_op == VMCTX_MAP_NONE)
		return 0;
	/*
	 * vmctx_apply_map() reads the context's own reply slot. Borrow it and
	 * put it back: the context is stopped in the monitor while this runs
	 * (the monitor is the caller), so nothing else is reading it.
	 */
	vmctx_apply_map_rep(target, vc, &m);
	return 0;
}

static long vmctx_ctl_take(struct task_struct *target, void __user *arg)
{
	struct vm_area_struct *vma;
	struct page *page = NULL;
	struct mm_struct *mm;
	struct vmctx_mem m;
	struct folio *folio;
	unsigned long addr;
	void *kaddr;
	long ret;

	if (copy_from_user(&m, arg, sizeof(m)))
		return -EFAULT;
	if (m.len < PAGE_SIZE)
		return -EINVAL;
	addr = m.addr & PAGE_MASK;

	vmctx_take_calls++;

	if (vmctx_syscall_holds(target, addr)) {
		vmctx_take_hold++;
		pr_warn_ratelimited("vmctx: TAKE 0x%lx pid %d: refused, held by in-flight syscall %llu (asked by %d %s)\n",
				    addr, task_pid_nr(target),
				    target->vmctx ? target->vmctx->sys_nr : 0,
				    task_pid_nr(current), current->comm);
		return -EBUSY;
	}

	mm = get_task_mm(target);
	if (!mm)
		return -ESRCH;

	mmap_read_lock(mm);
	/*
	 * FOLL_NOFAULT because asking for a page the context never touched must
	 * not create one: a zero page invented here becomes present, so the
	 * context never faults on that address again and reads this machine's
	 * zeros where its own memory belongs, for good. "Absent" is an answer,
	 * and the caller keeps its own copy when it hears it.
	 */
	ret = get_user_pages_remote(mm, addr, 1, FOLL_GET | FOLL_NOFAULT,
				    &page, NULL);
	if (ret != 1) {
		vmctx_take_absent++;
		ret = 0;		/* nothing here to take */
		goto out_unlock;
	}
	vma = vma_lookup(mm, addr);
	if (!vma) {
		put_page(page);
		vmctx_take_absent++;
		ret = 0;
		goto out_unlock;
	}

	folio = page_folio(page);

	/*
	 * The shared zero page, named -- because a refusal of it is the one
	 * -EBUSY in this file that does not mean "not yet".
	 *
	 * It gets here because something read the address without writing it,
	 * and a read fault on untouched anonymous memory installs
	 * empty_zero_page read-only. FOLL_NOFAULT will not create one, but it
	 * happily returns one already installed. The zero page is PG_reserved
	 * with a global reference count, so vmctx_folio_precheck() compares
	 * refs=2 against want=1(+1) and refuses, and will refuse the same way
	 * for ever; the caller's wait-and-ask-again then spends its whole
	 * second and gives up.
	 *
	 * Measured, and this is what the counters are for. Sixteen pg2 runs:
	 * take_refused=4004 and take_zeropage=4004, exactly equal -- every
	 * remaining pre-zap refusal in pg2 is this page, 1001 of them in each
	 * of the four runs that died. All 400 dumped refusals were pfn
	 * 0x23cc25, refcount:2 mapcount:0 mapping:NULL flags reserved, and that
	 * pfn is what /proc/self/pagemap reports for a page this machine has
	 * only read. (The earlier reading of this stall -- "refs=3 want=2(+1)
	 * mapcount=1 lru=1 anon=1" -- was a rate-limited line from a different
	 * refusal, not the one that repeated. Under pr_warn_ratelimited the
	 * line you see is not necessarily the line that happened.)
	 *
	 * Answering "nothing here" instead was tried, on the argument that a
	 * zero-page PTE holds no bytes of this address space and so ABSENT is
	 * the true answer. It is not an improvement: pg2 went 10/16 against the
	 * refusal's 12/16 on the same boot, and the deaths changed from a named
	 * FATAL in the monitor to the guest dying of its own accord. So the
	 * switch stays off and the refusal stays, with a line that says what it
	 * is. The repair belongs where the zero page was installed -- a guest
	 * fault serviced by the local kernel instead of by the monitor
	 * (AUDIT part III E2) -- not here.
	 */
	if (is_zero_page(page)) {
		vmctx_take_zeropage++;
		if (vmctx_take_zeropage_absent) {
			put_page(page);
			vmctx_take_absent++;
			ret = 0;
			goto out_unlock;
		}
		pr_warn_ratelimited("vmctx: TAKE 0x%lx: this is the shared zero page (pfn %lu): nothing has ever written the address on this side, it can never be claimed, and every retry will be refused the same way\n",
				    addr, page_to_pfn(page));
	}

	/*
	 * Ask first, destroy second.
	 *
	 * The order used to be the other way round, and the comment above this
	 * function argued for it: dropping the page first makes the range
	 * unreachable, so what is read afterwards is the page's final state.
	 * That argument is right about the copy and silent about the refusal.
	 * vmctx_folio_claim() may answer "not yet" -- and by then the page
	 * tables are gone, which for anonymous memory is everything the address
	 * had. The next take then answers "nothing here" about a page the
	 * program owned a millisecond ago, and a live stack becomes an empty
	 * page under a running thread. That is measured, not supposed: with the
	 * pagemap read either side of one refused take,
	 *
	 *     TAKE 0x7fffffffe000: 1 refusal(s); present before=1
	 *     after the first=0, take finally 0
	 *
	 * and it is how tests/pg3 died at rip=0 (AUDIT part VIII.3).
	 *
	 * PRINCIPLES.md §2 says -EBUSY means "wait and ask again". Waiting only
	 * works if the refusal leaves something to come back to. So every
	 * question that can be answered while the page is still mapped is asked
	 * here, before zap_page_range_single(), and a refusal at this point
	 * leaves the page exactly as it was found. The claim below is unchanged
	 * and still authoritative -- it is the atomic freeze that actually takes
	 * the page, and nothing here weakens it or hands over a page anything
	 * else can still reach.
	 */
	if (vmctx_take_claim_first) {
		/*
		 * A large folio has to stop being one before the question below
		 * has an answer at all. See vmctx_split_large().
		 */
		if (folio_test_large(folio)) {
			ret = vmctx_split_large(page, &folio, addr);
			if (ret) {
				put_page(page);
				vmctx_take_refused++;
				goto out_unlock;	/* still mapped, intact */
			}
		}
		ret = vmctx_folio_precheck(folio, 1, 1, "TAKE");
		if (ret) {
			put_page(page);
			vmctx_take_refused++;
			goto out_unlock;	/* the page is as it was */
		}
	}

	zap_page_range_single(vma, addr, PAGE_SIZE, NULL);

	/*
	 * The host PTE and its host-TLB shootdown reached only this context's mm.
	 * Kick every guest that may still hold the page in its own TLB, before the
	 * folio below is claimed and copied, so no store through a stale mapping
	 * outlives the copy. See vmctx_backend_invalidate().
	 */
	vmctx_backend_invalidate(target, addr);

	/*
	 * The page tables are gone; the references are the question. This is the
	 * path a monitor uses to give a page back to the machine running the
	 * program, and this side is the one performing the program's syscalls --
	 * so this is exactly where a buffer pinned by a read() that is still
	 * waiting on a device would be handed away while the device is about to
	 * write it. See vmctx_folio_may_move().
	 *
	 * Two references are ours: the FOLL_GET above and, for a file-backed
	 * page, the page cache's own. folio_expected_ref_count() accounts for the
	 * second, so only the first is added here.
	 */
	{
		long busy = vmctx_folio_claim(folio, 1);

		/*
		 * Past this point a refusal costs the page, so the wait
		 * PRINCIPLES.md §2 asks for happens here, where the page can
		 * still be waited for, rather than in userspace where it can
		 * only be waited for after it is gone. The precheck above has
		 * already refused everything that was true before the zap, so
		 * anything left is a reference taken in the window between the
		 * two -- a remote CPU's LRU batch, a get_user_pages() about to
		 * return -- and those end on their own. Bounded, small: the
		 * caller has its own second of asking behind this.
		 */
		if (busy) {
			int i;

			for (i = 0; busy && i < vmctx_take_retries; i++) {
				schedule_timeout_uninterruptible(1);
				busy = vmctx_folio_claim(folio, 1);
			}
			if (!busy)
				vmctx_take_regained++;
		}
		if (busy) {
			/*
			 * The one outcome that still costs a page, named so it
			 * can be counted rather than inferred. If this is ever
			 * non-zero the precheck is missing a case, and the case
			 * is in this line.
			 */
			vmctx_take_lost++;
			pr_warn_ratelimited("vmctx: TAKE 0x%lx: claim refused AFTER the zap (#%ld): refs=%d want=%d mapcount=%d pinned=%d lru=%d anon=%d -- the page is unmapped and was not served\n",
					    addr, vmctx_take_lost,
					    folio_ref_count(folio),
					    folio_expected_ref_count(folio),
					    folio_mapcount(folio),
					    folio_maybe_dma_pinned(folio),
					    folio_test_lru(folio),
					    folio_test_anon(folio));
			put_page(page);
			ret = busy;
			goto out_unlock;
		}
	}

	kaddr = kmap_local_page(page);
	ret = copy_to_user((void __user *)m.buf, kaddr, PAGE_SIZE) ?
		-EFAULT : PAGE_SIZE;
	vmctx_watch_check(addr, kaddr, "TAKE");
	kunmap_local(kaddr);
	vmctx_folio_release(folio, 1);
	put_page(page);
out_unlock:
	mmap_read_unlock(mm);
	mmput(mm);
	return ret;
}

/*
 * Take write permission away from a page of the context, leaving it readable.
 *
 * This is the state the "one writer" rule needs that removing a page cannot
 * express: both machines may hold a page at once as long as neither can write
 * it. Without it every page a syscall so much as *reads* has to be taken out of
 * the guest, and a page both sides read — a stack that syscall arguments point
 * into, while the guest is running on it — moves on nearly every fault.
 *
 * Only the page table is touched. Nothing here changes the VMA, so the guest's
 * own mprotect and the fault handler still see the permissions the program
 * asked for; a write lands as a protection fault, the monitor settles who owns
 * the page, and the page is made writable again by removing it and writing it
 * back (POKE forces write access, which allocates a fresh writable page). That
 * keeps this operation one-directional and means it cannot hand out write
 * permission to a page that was shared for copy-on-write.
 */
/*
 * May this page change machines right now?
 *
 * Removing the page tables is not the same as removing every way to reach the
 * page, and the difference is silent memory corruption. A syscall this side is
 * performing pins its buffers with get_user_pages() and keeps a struct page
 * reference across the whole call; a driver may hold a kernel mapping of it; a
 * device may already have its physical address in a descriptor. None of those
 * go away when the last PTE does. So a page handed to the other machine while
 * any of them is outstanding is a page that this machine can still write, after
 * the other machine has been told it owns it and has started writing it too.
 *
 * The case that makes it concrete: the program reads from disk into a buffer,
 * the call blocks waiting for the device, and while it is blocked the other
 * machine claims the buffer's page for a write of its own. The read completes
 * afterwards and the driver writes into memory that now belongs to the other
 * side, which never hears about it. Nothing in the page tables records that
 * this could happen and no test in the tree would see it.
 *
 * The kernel already tracks all of it, so this asks rather than guesses:
 *
 *   writeback  -- an outstanding write of this page; waited for, not refused,
 *                 because it finishes on its own and the wait is short.
 *   dma-pinned -- FOLL_PIN, which is exactly "a device may write this".
 *   mapped     -- a page table somewhere still reaches it, after the unmap.
 *   refcount   -- anything else holding it: a get_user_pages() reference taken
 *                 by a call that has not returned, a kernel mapping, an
 *                 in-flight I/O. Compared against what the page cache and its
 *                 mappings alone would account for, which is what the migration
 *                 code compares against for the same reason.
 *
 * -EBUSY, not a failure: the page cannot move *yet*. The caller waits and asks
 * again, which is the escalation lock this design needs, expressed where the
 * facts are rather than inferred from syscall boundaries in userspace.
 */
/*
 * Every way this function can refuse is named in the log, rate-limited.
 * The silent early returns cost a night: dma1 corrupted under #42 with zero
 * "page busy" lines, because the refusals that did fire were these two, and
 * they said nothing.
 */
static long vmctx_folio_claim(struct folio *folio, int extra_refs)
{
	folio_wait_writeback(folio);
	if (folio_maybe_dma_pinned(folio)) {
		pr_warn_ratelimited("vmctx: claim refused: dma-pinned refs=%d mapcount=%d\n",
				    folio_ref_count(folio),
				    folio_mapcount(folio));
		return -EBUSY;
	}
	if (folio_mapped(folio)) {
		pr_warn_ratelimited("vmctx: claim refused: still mapped %d time(s), refs=%d\n",
				    folio_mapcount(folio),
				    folio_ref_count(folio));
		return -EBUSY;
	}
	/*
	 * Take the page exclusively, rather than deciding that it looks
	 * takeable and copying it afterwards.
	 *
	 * folio_ref_freeze() sets the reference count to zero if and only if it
	 * is exactly what was expected, atomically, and fails if anyone else
	 * holds one. That is what mm/migrate.c does before moving a folio, and
	 * it answers -EAGAIN when it fails. Everything this function did before
	 * -- read the count, compare it, return, and let the caller copy -- was
	 * the same question asked with a window after it, in which a reference
	 * could be taken by anything.
	 *
	 * The count to expect is exactly what the page cache and the mappings
	 * account for, plus the references this caller holds. Not one more.
	 * Three kernels were spent establishing that, and the "one more" that
	 * kept appearing has now been named by their instruments read together:
	 *
	 *   #38 refused on the exact count with lru_add_drain() in front and
	 *       stalled the system -- but that drains only this CPU's LRU-add
	 *       batch, and the guest faults its pages in on a different CPU;
	 *   #40 reported, without refusing, one extra reference on every page it
	 *       looked at, all of them with lru=0 -- pages still sitting in some
	 *       CPU's LRU-add batch, whose batch slot IS a reference
	 *       folio_expected_ref_count() does not count;
	 *   #41 froze at exact+1 and refused 391k hand-overs in one suite run,
	 *       every one of them refs=2 want=1(+1) lru=1 -- pages already ON
	 *       the LRU, batch long drained, count exact, no holder at all.
	 *
	 * One population had a transient extra holder, the other did not, and a
	 * constant can satisfy only one of them. So: freeze at the exact count;
	 * if that fails on a folio not yet on the LRU, the likely holder is a
	 * remote CPU's batch, so drain them all -- the same call migration
	 * makes, and the _all is the point, the local drain was #38's mistake --
	 * and ask once more. A failure after that is a real holder: a DMA pin
	 * mid-flight, a get_user_pages() that has not returned, an in-flight
	 * read. -EBUSY, and the caller waits and asks again; the one thing it
	 * never does is hand over bytes anyway or answer with a page that is
	 * not the page (see vmctx/PRINCIPLES.md).
	 *
	 * Frozen only for the copy. The caller unfreezes it with
	 * vmctx_folio_release() once the bytes are out, because a frozen folio
	 * is one nothing else may touch at all -- which is the point, and also
	 * why it must not stay that way.
	 */
	if (!folio_ref_freeze(folio, folio_expected_ref_count(folio) +
				     extra_refs)) {
		static atomic_long_t seen;
		long n;

		if (!folio_test_lru(folio)) {
			lru_add_drain_all();
			if (folio_ref_freeze(folio,
					     folio_expected_ref_count(folio) +
					     extra_refs))
				return 0;
		}

		n = atomic_long_inc_return(&seen);
		if (n <= 8 || !(n % 1024))
			pr_warn("vmctx: page busy #%ld: refs=%d want=%d(+%d) mapcount=%d pinned=%d lru=%d anon=%d\n",
				n, folio_ref_count(folio),
				folio_expected_ref_count(folio), extra_refs,
				folio_mapcount(folio),
				folio_maybe_dma_pinned(folio),
				folio_test_lru(folio), folio_test_anon(folio));
		return -EBUSY;
	}
	return 0;
}

/*
 * The same question, asked while it is still free to ask.
 *
 * vmctx_folio_claim() is the answer -- an atomic freeze, taken only once the
 * mappings are gone, because a folio that is still mapped can be written
 * without ever taking a reference. But the only way to remove the mappings is
 * to destroy them, and for anonymous memory that is the whole of what the
 * address had. So a claim that refuses has already taken the page away from the
 * program in order to decide it may not be taken away from the program.
 *
 * This asks the same things beforehand, changing nothing:
 *
 *   writeback  -- waited for, exactly as the claim does; it ends on its own.
 *   dma-pinned -- a device may write this page. True before the zap and after
 *                 it, so it costs nothing to hear it first.
 *   mappings   -- the caller says how many its own unmap will remove
 *                 (@max_maps, or -1 for "all of them"). A folio mapped more
 *                 times than that is one the claim will refuse for "still
 *                 mapped", which is knowable now.
 *   references -- the count the claim will compare against, compared now.
 *                 folio_expected_ref_count() counts one per page-table mapping,
 *                 and an unmap drops both a mapping and its reference, so the
 *                 *difference* between the real count and the expected one is
 *                 what survives the unmap. Zero now means zero after, unless
 *                 something takes a reference in between -- which is a race,
 *                 not a state, and is what the retry in the caller is for.
 *
 * Large folios are the one case this cannot answer: the caller's unmap covers
 * one page of them, and how many mappings that removes depends on how the folio
 * is mapped. They are counted (vmctx_take_large) and left to the claim, which
 * is exactly the behaviour they had before this existed.
 *
 * Upstream says the same thing about the same helper, in
 * folio_expected_ref_count()'s own comment: it can be used "to detect
 * unexpected references early (for example, if it makes sense to even lock the
 * folio and unmap it)". This is that.
 *
 * Returns 0 if the hand-over may proceed, -EBUSY -- "not yet", never "no" --
 * with the page untouched otherwise.
 */
static long vmctx_folio_precheck(struct folio *folio, int max_maps,
				 int extra_refs, const char *what)
{
	int refs, want;

	folio_wait_writeback(folio);

	if (folio_maybe_dma_pinned(folio)) {
		pr_warn_ratelimited("vmctx: %s refused before the unmap: dma-pinned refs=%d mapcount=%d\n",
				    what, folio_ref_count(folio),
				    folio_mapcount(folio));
		return -EBUSY;
	}

	if (folio_test_large(folio)) {
		/*
		 * Still not answerable here. The TAKE path no longer arrives
		 * with one -- vmctx_split_large() has made it into folios this
		 * can answer for -- so this counts what is left: TAKEOBJ, and
		 * TAKE with vmctx_take_split_large switched off.
		 */
		vmctx_precheck_large++;
		return 0;	/* not answerable here; the claim still answers */
	}

	if (max_maps >= 0 && folio_mapcount(folio) > max_maps) {
		pr_warn_ratelimited("vmctx: %s refused before the unmap: mapped %d time(s), more than the %d this unmap removes, refs=%d\n",
				    what, folio_mapcount(folio), max_maps,
				    folio_ref_count(folio));
		return -EBUSY;
	}

	refs = folio_ref_count(folio);
	want = folio_expected_ref_count(folio) + extra_refs;
	if (refs != want) {
		int was = refs;

		/*
		 * The holder is most likely some CPU's folio batch, whose slot
		 * is a reference folio_expected_ref_count() does not count.
		 * Drain them all -- _all, because the guest faults its pages in
		 * on whichever CPU it was running on, and draining only this
		 * one was kernel #38's mistake -- and ask again.
		 *
		 * This used to happen only for a folio that was NOT on the LRU,
		 * on the reasoning that the batch in question is the LRU-*add*
		 * batch, which is the one a folio sits in before it is on the
		 * LRU. That reasoning excluded the case actually measured:
		 * pg2 stalls on 0x7ffff7ff6000 with refs=3 want=2(+1)
		 * mapcount=1 lru=1 anon=1 for a full second and then gives up.
		 * lru=1, so the drain never ran; and folio_activate(),
		 * lru_deactivate_file(), lru_lazyfree() and the mlock batches
		 * all hold a reference to a folio that is already on the LRU.
		 * So the condition was the instrument's blind spot rather than
		 * a property of the folio, and it is gone.
		 */
		if (vmctx_take_drain_always || !folio_test_lru(folio)) {
			lru_add_drain_all();
			refs = folio_ref_count(folio);
			want = folio_expected_ref_count(folio) + extra_refs;
			if (refs == want) {
				vmctx_take_drained++;
				return 0;
			}
		}

		/*
		 * Still held. Name the holder rather than describing it:
		 * dump_page() prints the flags, the mapping, the index, the
		 * memcg and the type, which is what tells a swapcache
		 * reference from an isolated-for-migration one from a pin.
		 * Counted down so it costs nothing after the first few, and
		 * re-armable by writing vmctx_take_dump again.
		 */
		pr_warn_ratelimited("vmctx: %s refused before the unmap: refs=%d want=%d(+%d) (was %d before the drain) mapcount=%d lru=%d anon=%d swapcache=%d writeback=%d dirty=%d mlocked=%d unevictable=%d ksm=%d private=%d order=%d\n",
				    what, refs, want, extra_refs, was,
				    folio_mapcount(folio),
				    folio_test_lru(folio),
				    folio_test_anon(folio),
				    folio_test_swapcache(folio),
				    folio_test_writeback(folio),
				    folio_test_dirty(folio),
				    folio_test_mlocked(folio),
				    folio_test_unevictable(folio),
				    folio_test_ksm(folio),
				    folio_test_private(folio),
				    folio_order(folio));
		if (vmctx_take_dump > 0) {
			vmctx_take_dump--;
			dump_page(&folio->page, "vmctx: refused before the unmap");
		}
		return -EBUSY;
	}
	return 0;
}

/* Give back what vmctx_folio_claim() froze, once the bytes are out. */
static void vmctx_folio_release(struct folio *folio, int extra_refs)
{
	folio_ref_unfreeze(folio, folio_expected_ref_count(folio) +
				  extra_refs);
}

/*
 * Take a page out of an *address space*, rather than out of one context.
 *
 * This is the operation the model always needed and never had, and its absence
 * is why "the page has been handed over" was never quite true.
 *
 * A guest address space on this machine is one backing object mapped into
 * several contexts, each with page tables of its own. VMCTX_CTL_TAKE works on
 * one of those: it pins the page, zaps that one VMA, and copies the bytes
 * afterwards out of the pinned page. Everything that makes the copy safe is
 * therefore missing. The other contexts still map the same physical page and
 * are not stopped, so the bytes can change under the copy; nothing removes the
 * page from the object, so any of them can fault it straight back in; and the
 * TLB flush that comes with the zap reaches only the CPUs of that one mm, so a
 * sibling's vCPU keeps a translation to a page it has been told it lost.
 * Userspace then issues one of these per context and punches the object
 * separately, which turns a single hand-over into a sequence with a gap at
 * every join.
 *
 * There is no way to close those gaps from above, because each one is a window
 * inside an operation userspace cannot enter. So the hand-over becomes one
 * operation here:
 *
 *   - lock the folio. From this point a fault anywhere in any mm that wants
 *     this offset blocks in shmem_fault() on exactly this lock, so no context
 *     can acquire the page while the hand-over is in progress.
 *   - unmap_mapping_range() over the object's mapping, which removes the page
 *     from *every* mm that has it, with each mm's own TLB flush, rather than
 *     from the one this call happens to name.
 *   - copy, with no mapping left anywhere and every would-be faulter parked.
 *   - remove it from the object, so this side can no longer answer for it and
 *     the next touch has to ask the machine that now holds it.
 *
 *   - and remove it from the object WITHOUT LETTING GO OF THE LOCK, which is
 *     what closes the last window.
 *
 * That window used to be here, named rather than hidden: the folio lock had to
 * be dropped before shmem_truncate_range(), because that takes it itself, and a
 * context parked in shmem_fault() could be woken in the gap, map the page,
 * write it, and have the truncate discard the write. It was described as a
 * strictly smaller hole than the userspace loop it replaced, and it was -- but
 * it was still a hole, and a guest's store falling into it is silent: the
 * source is handed the copy taken before the store, and hands that copy back
 * later. tests/hx1.c reproduces it, four threads writing their own words in one
 * page while a forwarded read(2) keeps wanting that page.
 *
 * filemap_remove_folio() is what shmem_truncate_range() would have reached, and
 * it REQUIRES the folio locked -- so the removal happens with every would-be
 * faulter still parked, and they wake to a hole and go and ask the machine that
 * owns the page. There is no longer a moment in this operation when the page is
 * both absent from every page table and still available in the object.
 *
 * Returns PAGE_SIZE with the bytes, 0 if the address space does not have the
 * page (not an error: the caller keeps whatever it already had), or -ENOENT if
 * the range is not object-backed, which is the caller's signal to fall back.
 */
static long vmctx_ctl_takeobj(struct task_struct *target, void __user *arg)
{
	struct address_space *mapping;
	struct vm_area_struct *vma;
	struct mm_struct *mm;
	struct vmctx_mem m;
	struct folio *folio;
	struct file *file;
	unsigned long addr;
	struct inode *inode;
	pgoff_t index;
	loff_t pos;
	void *kaddr;
	long ret;

	if (copy_from_user(&m, arg, sizeof(m)))
		return -EFAULT;
	if (m.len < PAGE_SIZE)
		return -EINVAL;
	addr = m.addr & PAGE_MASK;

	if (vmctx_syscall_holds(target, addr)) {
		pr_warn_ratelimited("vmctx: TAKEOBJ 0x%lx pid %d: refused, held by in-flight syscall %llu (asked by %d %s)\n",
				    addr, task_pid_nr(target),
				    target->vmctx ? target->vmctx->sys_nr : 0,
				    task_pid_nr(current), current->comm);
		return -EBUSY;
	}

	mm = get_task_mm(target);
	if (!mm)
		return -ESRCH;

	mmap_read_lock(mm);
	vma = vma_lookup(mm, addr);
	/*
	 * Only a mapping of the backing object can be handed over this way. A
	 * private or anonymous range has no object to remove the page from, and
	 * answering for one here would hand over a copy while leaving the
	 * original in place -- two holders, which is the state all of this
	 * exists to prevent.
	 */
	if (!vma || !vma->vm_file || !(vma->vm_flags & VM_SHARED)) {
		mmap_read_unlock(mm);
		mmput(mm);
		return -ENOENT;
	}
	file = get_file(vma->vm_file);
	index = linear_page_index(vma, addr);
	mmap_read_unlock(mm);
	mmput(mm);

	mapping = file->f_mapping;
	inode = file->f_inode;
	if (!shmem_mapping(mapping)) {
		fput(file);
		return -ENOENT;
	}
	pos = (loff_t)index << PAGE_SHIFT;

	folio = filemap_lock_folio(mapping, index);
	if (IS_ERR(folio)) {
		fput(file);
		return 0;		/* the address space does not have it */
	}
	/*
	 * A large folio covers more than the page being handed over, and
	 * removing it would take memory the caller never asked about. Refused
	 * rather than guessed at; the caller falls back to the per-context
	 * path, which is no worse than what it did before this existed.
	 */
	if (folio_test_large(folio)) {
		folio_unlock(folio);
		folio_put(folio);
		fput(file);
		return -ENOENT;
	}

	/*
	 * DMA pin is the one writer the folio lock cannot stop; check it before
	 * unmapping (a cheap early-out) and again after (the check that counts).
	 */
	folio_wait_writeback(folio);
	if (folio_maybe_dma_pinned(folio)) {
		vmctx_takeobj_refused++;
		folio_unlock(folio);
		folio_put(folio);
		fput(file);
		return -EBUSY;
	}

	unmap_mapping_range(mapping, pos, PAGE_SIZE, 0);

	/*
	 * unmap_mapping_range flushed each mapper's host TLB, but not the guests'
	 * own ASID-tagged TLBs. Kick every running guest out before the folio is
	 * examined and copied below, so a store made through a stale mapping in
	 * the window since the unmap is captured in the folio we still hold rather
	 * than lost. See vmctx_backend_invalidate().
	 */
	vmctx_backend_invalidate(target, addr);

	/*
	 * After the unmap under the lock no CPU has a PTE and none can make one
	 * (a fault for this offset blocks in shmem_fault on the lock we hold),
	 * so references held by parked faulters cannot write and no freeze is
	 * needed -- the lock is the exclusion. The freeze this used to do was a
	 * livelock under multi-context contention (the count is never the
	 * expected value). The only writer left is a device pin, which is a
	 * refcount not a mapping and survives the unmap, so it is re-checked
	 * here, the last instant one can exist.
	 */
	if (folio_mapped(folio) || folio_maybe_dma_pinned(folio)) {
		pr_warn_ratelimited("vmctx: TAKEOBJ 0x%lx: after unmap under the lock, mapped=%d dma_pinned=%d refs=%d; refusing\n",
				    addr, folio_mapcount(folio),
				    folio_maybe_dma_pinned(folio),
				    folio_ref_count(folio));
		vmctx_takeobj_lost++;
		folio_unlock(folio);
		folio_put(folio);
		fput(file);
		return -EBUSY;
	}

	kaddr = kmap_local_folio(folio, 0);
	ret = copy_to_user((void __user *)m.buf, kaddr, PAGE_SIZE) ?
		-EFAULT : PAGE_SIZE;
	vmctx_watch_check(addr, kaddr, "TAKEOBJ");
	/* An all-zero capture names the folio's maker (see vmctx_shmem_alloced). */
	if (ret == PAGE_SIZE && !memchr_inv(kaddr, 0, PAGE_SIZE))
		vmctx_shm_explain_zero(mapping, index, addr, target);
	kunmap_local(kaddr);

	/*
	 * And OUT OF THE OBJECT WHILE STILL HOLDING ITS LOCK. This is the whole
	 * of the fix, and the window it closes was named in this function's own
	 * comment from the day it was written:
	 *
	 *   "the folio lock has to be dropped before shmem_truncate_range(),
	 *    which takes it itself. A context that was blocked in shmem_fault()
	 *    can be woken in that gap, map the page and write it, and the
	 *    truncate then discards the write."
	 *
	 * Every context that wants this offset is parked in shmem_fault() on
	 * this lock right now. Unlocking first wakes one of them onto a page
	 * that is still in the object -- it maps it, writes it, and the truncate
	 * that follows throws that write away. The guest stored a value and the
	 * value is gone, with nothing anywhere reporting it: the source is given
	 * the copy taken before the store, and the page it hands back later is
	 * that copy. tests/hx1.c is that, minimised: a thread stores to its own
	 * eight bytes, reads them straight back, and gets a value thousands of
	 * rounds old.
	 *
	 * filemap_remove_folio() is the operation shmem_truncate_range() would
	 * have reached anyway, minus the lock dance: it requires the folio
	 * LOCKED, which is precisely the property that makes it safe here.
	 * After it the offset is a hole, every parked faulter wakes to find
	 * nothing, and the fault goes where it must -- to the machine that now
	 * owns the page.
	 *
	 * shmem_recalc_inode() is the accounting shmem_undo_range() does after
	 * removing a folio, and skipping it would leak the inode's alloced count
	 * once per hand-over. It has no public declaration because nothing
	 * outside mm/shmem.c has needed it; this is in-tree code, so it is
	 * declared above rather than duplicated.
	 *
	 * ZERO, not -1, and the difference is a block leaked per hand-over.
	 *
	 * shmem_recalc_inode() does not take "how many pages went away" -- it
	 * DERIVES that, from
	 *
	 *     freed = alloced - swapped - i_mapping->nrpages
	 *
	 * and only then unaccounts the blocks. Every call site in mm/shmem.c
	 * passes 0 for `alloced` after removing a folio for exactly this reason
	 * (shmem_undo_range: `shmem_recalc_inode(inode, 0, -nr_swaps_freed)`),
	 * because filemap_remove_folio() has already dropped nrpages.
	 *
	 * Passing -1 subtracts the page a second time: alloced falls by one AND
	 * nrpages falls by one, so `freed` computes 0, shmem_inode_unacct_blocks()
	 * is never reached, and inode->i_blocks keeps the block for ever. The
	 * kernel says so itself at teardown --
	 *
	 *   WARNING: mm/shmem.c:1430 at shmem_evict_inode+0xf8/0x2b0
	 *
	 * which is WARN_ON(inode->i_blocks) as the backing memfd is finally
	 * evicted, one leaked block per punched hand-over. It also hardcoded a
	 * single page for a folio that may be larger; deriving the count cannot
	 * get that wrong either.
	 *
	 * The order is copy -> remove -> unlock, the lock spanning all of it.
	 */
	if (ret == PAGE_SIZE && vmctx_takeobj_punch) {
		vmctx_takeobj_gaps++;
		filemap_remove_folio(folio);
		shmem_recalc_inode(inode, 0, 0);
	}

	folio_unlock(folio);
	folio_put(folio);
	fput(file);
	return ret;
}

/* Only the object's shared mappings -- the guest contexts -- are the ones
 * promised read-only; a private mapping of the folio, if there were one, has no
 * such promise to keep. This is invalid_mkclean_vma() by another name. */
static bool vmctx_wrprotect_invalid_vma(struct vm_area_struct *vma, void *arg)
{
	return !(vma->vm_flags & VM_SHARED);
}

/*
 * The per-mapping half of the object-wide protect. It does exactly what
 * vmctx_ctl_protect() does to a single context, and no more: clear the write
 * bit, keep the dirty bit, flush.
 *
 * folio_mkclean() was the obvious primitive and it is the wrong one. Besides
 * write-protecting it also pte_mkclean()s, and a *clean* write-protected shmem
 * page takes its next write fault down a path the guest does not come back
 * from: pg1's stamper wrote its counter once, the source protected the page,
 * and the stamper never advanced it again -- "the source holding a page the
 * guest can no longer reach". Preserving the dirty bit, as ptep_set_wrprotect()
 * does, is what keeps that write fault on the road back to the coherence layer.
 */
static bool vmctx_wrprotect_one(struct folio *folio, struct vm_area_struct *vma,
				unsigned long address, void *arg)
{
	DEFINE_FOLIO_VMA_WALK(pvmw, folio, vma, address, PVMW_SYNC);
	int *count = arg;

	while (page_vma_mapped_walk(&pvmw)) {
		pte_t old, new;

		if (!pvmw.pte)
			continue;	/* huge folios are refused upstream */
		old = ptep_get(pvmw.pte);
		if (!pte_present(old) || !pte_write(old))
			continue;
		switch (vmctx_protobj_mode) {
		default:
		case 0:	/* keep dirty, no not-present window */
			ptep_set_wrprotect(vma->vm_mm, pvmw.address, pvmw.pte);
			flush_tlb_page(vma, pvmw.address);
			break;
		case 1:	/* keep dirty, with window (ptep_clear_flush) */
			old = ptep_clear_flush(vma, pvmw.address, pvmw.pte);
			set_pte_at(vma->vm_mm, pvmw.address, pvmw.pte,
				   pte_wrprotect(old));
			break;
		case 2:	/* clear dirty, with window (== folio_mkclean) */
			old = ptep_clear_flush(vma, pvmw.address, pvmw.pte);
			set_pte_at(vma->vm_mm, pvmw.address, pvmw.pte,
				   pte_mkclean(pte_wrprotect(old)));
			break;
		case 3:	/* clear dirty, no window */
			new = pte_mkclean(pte_wrprotect(old));
			set_pte_at(vma->vm_mm, pvmw.address, pvmw.pte, new);
			flush_tlb_page(vma, pvmw.address);
			break;
		}
		(*count)++;
		vmctx_protobj_prot++;
	}
	return true;
}

/*
 * Write-protect every mapping of the folio at once, holding its lock so no new
 * mapping can appear mid-walk (a faulter blocks in shmem_fault on this lock).
 * The flush in vmctx_wrprotect_one() is a broadcast shootdown; it forces any
 * guest running a mapping to VMEXIT, and the backend re-enters with the guest
 * TLB flushed, so once this returns no context can write the page until it
 * faults. That is the atomicity the per-context loop cannot give. Returns the
 * number of mappings actually write-protected.
 */
static int vmctx_folio_wrprotect_all(struct folio *folio)
{
	int count = 0;
	struct rmap_walk_control rwc = {
		.arg = &count,
		.rmap_one = vmctx_wrprotect_one,
		.invalid_vma = vmctx_wrprotect_invalid_vma,
	};

	rmap_walk(folio, &rwc);
	return count;
}

/*
 * The read-claim counterpart of vmctx_ctl_takeobj: write-protect the object
 * folio in EVERY mapping at once, atomically, then hand back its bytes. The
 * per-context VMCTX_CTL_PROTECT loop in userspace protects one sibling at a
 * time, and between protecting sibling A and sibling B, B can still write the
 * page the source has just been promised nobody will -- the both-hold lost
 * write. Doing it under the folio lock, over the whole rmap, closes that window.
 */
static long vmctx_ctl_protectobj(struct task_struct *target, void __user *arg)
{
	struct address_space *mapping;
	struct vm_area_struct *vma;
	struct mm_struct *mm;
	struct vmctx_mem m;
	struct folio *folio;
	struct file *file;
	unsigned long addr;
	pgoff_t index;
	void *kaddr;
	long ret;

	if (copy_from_user(&m, arg, sizeof(m)))
		return -EFAULT;
	if (m.len < PAGE_SIZE)
		return -EINVAL;
	addr = m.addr & PAGE_MASK;

	/*
	 * The same exclusion vmctx_ctl_takeobj() makes, for the same reason: an
	 * in-flight forwarded syscall is a writer this side is about to admit to
	 * the page after the copy below. Its result is written into guest memory
	 * with FOLL_WRITE, which re-grants write on the shared vma through a
	 * fault the protect cannot stop, so a page read-claimed underneath it
	 * would leave this side one write newer than the copy the source was
	 * given -- the read-claim's version of the lost write. Refuse; the caller
	 * answers ABSENT and the source keeps its own copy until the hold ends.
	 */
	if (vmctx_syscall_holds(target, addr)) {
		pr_warn_ratelimited("vmctx: PROTECTOBJ 0x%lx pid %d: refused, held by in-flight syscall %llu (asked by %d %s)\n",
				    addr, task_pid_nr(target),
				    target->vmctx ? target->vmctx->sys_nr : 0,
				    task_pid_nr(current), current->comm);
		return -EBUSY;
	}

	mm = get_task_mm(target);
	if (!mm)
		return -ESRCH;

	mmap_read_lock(mm);
	vma = vma_lookup(mm, addr);
	if (!vma || !vma->vm_file || !(vma->vm_flags & VM_SHARED)) {
		mmap_read_unlock(mm);
		mmput(mm);
		return -ENOENT;		/* not object-backed */
	}
	file = get_file(vma->vm_file);
	index = linear_page_index(vma, addr);
	mmap_read_unlock(mm);
	mmput(mm);

	mapping = file->f_mapping;
	if (!shmem_mapping(mapping)) {
		fput(file);
		return -ENOENT;
	}

	folio = filemap_lock_folio(mapping, index);
	if (IS_ERR(folio)) {
		fput(file);
		return 0;		/* the object does not have it */
	}
	if (folio_test_large(folio)) {
		folio_unlock(folio);
		folio_put(folio);
		fput(file);
		return -ENOENT;
	}

	/*
	 * A device pin is the writer the write-protect below cannot reach: DMA
	 * writes bypass the PTE the folio lock and folio_mkclean() act on, so a
	 * pinned folio could be advanced after the copy and the source given a
	 * page one write old. vmctx_ctl_takeobj() refuses it for the same reason;
	 * so does this. Nothing is being removed, so no writeback wait is needed
	 * -- writeback reads the folio, it does not write it.
	 */
	if (folio_maybe_dma_pinned(folio)) {
		folio_unlock(folio);
		folio_put(folio);
		fput(file);
		return -EBUSY;
	}

	/*
	 * A folio in the object that no context maps is reported absent, exactly
	 * as the per-context loop this replaces did: it protected on VMR_PG_SIZE
	 * ("this context had it and it is now read-only") and never on the
	 * kernel's "not present here", so a page nobody mapped was a failure and
	 * the source fell back to its own copy. Widening that to success measured
	 * worse in the per-context form for reasons that were never pinned down;
	 * the single thing changing here is the atomicity of the protect, so that
	 * predicate is held fixed. folio_mapped() under the folio lock is the
	 * object-wide form of "some context had it".
	 */
	if (!folio_mapped(folio)) {
		folio_unlock(folio);
		folio_put(folio);
		fput(file);
		return 0;
	}

	/*
	 * Write-protect the folio in every mapping at once -- clearing the write
	 * bit only, keeping dirty, exactly as the per-context VMCTX_CTL_PROTECT
	 * does but over the whole rmap under the folio lock. See
	 * vmctx_folio_wrprotect_all() and vmctx_wrprotect_one() for why this is
	 * not folio_mkclean().
	 */
	vmctx_folio_wrprotect_all(folio);

	/*
	 * The per-mapping shootdown above clears the host TLBs but not a running
	 * guest's own ASID-tagged TLB, which may still hold the page WRITABLE.
	 * Force every guest out so its next entry reloads the now-read-only
	 * mapping; without this the "nothing can have changed the bytes since the
	 * protect" claim below is false -- a guest keeps writing through the stale
	 * writable entry. See vmctx_backend_invalidate().
	 */
	vmctx_backend_invalidate(target, addr);

	/*
	 * And read the bytes under the same lock: nothing can have changed them
	 * since the protect above, so this copy is the page as of the protect,
	 * not one write old. A caller that only needs the protection (not the
	 * bytes) passes buf == 0 and gets PAGE_SIZE without a copy -- the atomic
	 * downgrade is the whole product, and the folio stays where it is.
	 */
	if (m.buf) {
		kaddr = kmap_local_folio(folio, 0);
		ret = copy_to_user((void __user *)m.buf, kaddr, PAGE_SIZE) ?
			-EFAULT : PAGE_SIZE;
		vmctx_watch_check(addr, kaddr, "PROTECTOBJ");
		kunmap_local(kaddr);
	} else {
		ret = PAGE_SIZE;
	}

	folio_unlock(folio);
	folio_put(folio);
	fput(file);
	return ret;
}

static long vmctx_ctl_protect(struct task_struct *target, void __user *arg)
{
	struct vm_area_struct *vma;
	struct mm_struct *mm;
	struct vmctx_mem m;
	unsigned long addr;
	spinlock_t *ptl;
	pte_t *ptep;
	long ret = 0;

	if (copy_from_user(&m, arg, sizeof(m)))
		return -EFAULT;
	if (m.len < PAGE_SIZE)
		return -EINVAL;
	addr = m.addr & PAGE_MASK;

	mm = get_task_mm(target);
	if (!mm)
		return -ESRCH;

	mmap_read_lock(mm);
	vma = vma_lookup(mm, addr);
	if (!vma) {
		ret = -ENOENT;
		goto out;
	}
	ptep = get_locked_pte(mm, addr, &ptl);
	if (!ptep) {
		ret = -ENOMEM;
		goto out;
	}
	if (!pte_present(ptep_get(ptep))) {
		/*
		 * Nothing mapped, so nothing to take away. Not an error: the
		 * page is already unreachable, which is stronger than
		 * read-only, and the caller wants it no more writable than it
		 * asked for.
		 */
		pte_unmap_unlock(ptep, ptl);
		goto out;
	}
	ptep_set_wrprotect(mm, addr, ptep);
	pte_unmap_unlock(ptep, ptl);
	/*
	 * The write bit is only gone once no CPU still has the old translation.
	 * Skipping this is invisible on the machine that ran the ioctl and lets
	 * another CPU keep writing a page it has been told it may not.
	 */
	flush_tlb_mm_range(mm, addr, addr + PAGE_SIZE, PAGE_SHIFT, false);
	/* Host flush done; kick the guest's own TLB the same way. See
	 * vmctx_backend_invalidate(). */
	vmctx_backend_invalidate(target, addr);
	ret = PAGE_SIZE;
out:
	mmap_read_unlock(mm);
	mmput(mm);
	return ret;
}

static long vmctx_ctl_mem(struct task_struct *target, void __user *arg, bool write)
{
	struct vmctx_mem m;
	void *kbuf;
	long ret;
	size_t len;

	if (copy_from_user(&m, arg, sizeof(m)))
		return -EFAULT;
	len = min_t(size_t, m.len, PAGE_SIZE);
	if (!len)
		return 0;

	kbuf = kmalloc(len, GFP_KERNEL);
	if (!kbuf)
		return -ENOMEM;

	if (write) {
		if (copy_from_user(kbuf, (void __user *)m.buf, len)) {
			ret = -EFAULT;
			goto out;
		}
		/*
		 * FOLL_FORCE, for the same reason ptrace uses it: a monitor has
		 * to be able to write pages the guest maps read-only — its text
		 * and its read-only data. Without this, installing the contents
		 * of a demand-paged library silently writes nothing and the
		 * guest reads zeros.
		 */
		ret = access_process_vm(target, m.addr, kbuf, len,
					FOLL_WRITE | FOLL_FORCE);
	} else {
		ret = access_process_vm(target, m.addr, kbuf, len, 0);
		if (ret > 0 && copy_to_user((void __user *)m.buf, kbuf, ret))
			ret = -EFAULT;
	}
out:
	kfree(kbuf);
	return ret;
}

long vmctx_ctl_impl(pid_t pid, unsigned int cmd, void __user *arg)
{
	struct task_struct *target;
	struct vmctx_task *vc;
	long ret = 0;

	target = find_get_task_by_vpid(pid);
	if (!target)
		return -ESRCH;
	if (!ptrace_may_access(target, PTRACE_MODE_ATTACH_REALCREDS)) {
		ret = -EPERM;
		goto out_put;
	}

	if (cmd == VMCTX_CTL_ADOPT) {
		struct vmctx_task *nvc = kzalloc(sizeof(*nvc), GFP_KERNEL);

		if (!nvc) {
			ret = -ENOMEM;
			goto out_put;
		}
		kref_init(&nvc->kref);
		/*
		 * Its faults go to the monitor, and they have to.
		 *
		 * A service context holds the program's real address space, so
		 * source-side reads of it are right by construction. Its
		 * *writes* are not: the program's stores happen on the machine
		 * running its instructions, into memory this task has never
		 * seen. So a page this side touches may be one the other side
		 * has since written, and the only thing that can supply it is
		 * the monitor.
		 *
		 * Without this the context faults, no one is asked, and
		 * do_anonymous_page() invents a zero page -- which is how a
		 * nine-element iovec the guest had just built on its own stack
		 * was read here as nine zero lengths.
		 */
		nvc->flags = VMCTX_FLAG_SERVICE | VMCTX_FLAG_REDIRECT_FAULT;
		nvc->backend = NULL;	/* a service context never enters one */
		init_waitqueue_head(&nvc->ev_wq);
		init_waitqueue_head(&nvc->reply_wq);
		init_waitqueue_head(&nvc->sys_wq);
		spin_lock_init(&nvc->ev_lock);
		spin_lock_init(&nvc->sys_pages_lock);
		get_task_struct(current);
		nvc->monitor = current;

		task_lock(target);
		if (target->vmctx) {
			task_unlock(target);
			put_task_struct(current);
			kfree(nvc);
			ret = -EEXIST;
			goto out_put;
		}
		target->vmctx = nvc;
		task_unlock(target);

		/*
		 * Armed last. From here the task's next return to user mode
		 * parks it (see vmctx_guest_step), so it holds the program
		 * without ever running it.
		 */
		nvc->active = 1;
		nvc->noted_mm = target->mm;
		vmctx_mm_note(target->mm);
		atomic_inc(&vmctx_nr_active);
		pr_info("vmctx: pid %d adopted as a service context by pid %d\n",
			task_pid_nr(target), task_pid_nr(current));
		ret = 0;
		goto out_put;
	}

	task_lock(target);
	vc = target->vmctx;
	if (vc)
		kref_get(&vc->kref);
	task_unlock(target);
	if (!vc) {
		ret = -EINVAL;		/* not a VM context */
		goto out_put;
	}
	if (vc->dead) {
		ret = -ESRCH;
		goto out_vc;
	}
	/*
	 * Every command except ATTACH requires that we are the monitor. Any
	 * thread of the monitoring process counts: a monitor typically has one
	 * thread blocked in WAIT while another services page requests.
	 */
	if (cmd != VMCTX_CTL_ATTACH && !(vc->monitor &&
					 same_thread_group(vc->monitor, current))) {
		ret = -EPERM;
		/*
		 * out_vc, not out_put: a reference to vc was taken above and
		 * out_put skips the kref_put. This leaked one reference on
		 * every refused call, so a context that was probed by anything
		 * other than its own monitor was never freed.
		 */
		goto out_vc;
	}

	switch (cmd) {
	case VMCTX_CTL_ATTACH:
		if (vc->monitor && vc->monitor != current) {
			ret = -EBUSY;
		} else if (!vc->monitor) {
			get_task_struct(current);	/* keep it valid while attached */
			vc->monitor = current;
			/* Release a context waiting in VMCTX_FLAG_WAIT_MONITOR. */
			wake_up(&vc->reply_wq);
		}
		break;
	case VMCTX_CTL_DETACH:
		vmctx_drop_monitor(vc);
		wake_up(&vc->reply_wq);		/* unblock the context */
		break;
	case VMCTX_CTL_WAIT: {
		struct vmctx_event ev;

		for (;;) {
			ret = wait_event_interruptible(vc->ev_wq,
						       vc->ev_pending || vc->dead);
			if (ret)
				goto wait_out;
			if (vc->dead) {
				ret = -ESRCH;	/* gone; stop monitoring */
				goto wait_out;
			}
			/*
			 * Claim it under the lock, and copy from the claimed
			 * snapshot rather than from the context.
			 *
			 * ev_pending is re-tested here because the waiter may
			 * have withdrawn the event between the wake-up and now
			 * -- that is the whole point of the lock, and taking an
			 * event that has been withdrawn is how the same syscall
			 * gets run twice. A withdrawn event simply means going
			 * back to sleep.
			 */
			spin_lock(&vc->ev_lock);
			if (vc->ev_pending) {
				ev = vc->ev;
				vc->ev_taken = 1;   /* a reply to this is owed */
				spin_unlock(&vc->ev_lock);
				break;
			}
			spin_unlock(&vc->ev_lock);
		}
		if (copy_to_user(arg, &ev, sizeof(ev))) {
			ret = -EFAULT;
			spin_lock(&vc->ev_lock);
			vc->ev_taken = 0;
			spin_unlock(&vc->ev_lock);
		}
wait_out:
		break;
	}
	case VMCTX_CTL_RESUME: {
		struct vmctx_reply rep;

		if (copy_from_user(&rep, arg, sizeof(rep))) {
			ret = -EFAULT;
			break;
		}
		if (vc->ev_abandoned) {
			/*
			 * This answers an event whose waiter gave up (a signal
			 * arrived). Accepting it would leave a reply sitting in
			 * the context for the next, unrelated event.
			 *
			 * Drop the reply and nothing else. The context has very
			 * likely re-published by now — it retries as soon as the
			 * signal is delivered — and clearing ev_pending here
			 * unpublishes that new event: the context then waits for
			 * a reply to something the monitor will never be told
			 * about, and the monitor waits for an event that has
			 * already been taken off the queue. Both sides sleep,
			 * neither is at fault, and the run simply stops.
			 *
			 * Nothing sets ev_abandoned any more: a waiter may only
			 * withdraw an event the monitor has not taken, and the
			 * two are decided under ev_lock. This is kept as the
			 * check that says so, and it says it out loud, because
			 * an event that was both taken and abandoned means the
			 * syscall has run twice.
			 */
			pr_warn("vmctx: pid %d: a reply arrived for an event its waiter had abandoned; the syscall may have run twice\n",
				task_pid_nr(target));
			vc->ev_abandoned = 0;
			vc->released     = 1;
			break;
		}
		vc->reply = rep;
		vc->ev_pending    = 0;
		vc->reply_pending = 1;
		vc->released      = 1;	/* also releases a restoring context */
		wake_up(&vc->reply_wq);
		break;
	}
	case VMCTX_CTL_GETREGS: {
		struct vmctx_uregs u;

		uregs_from_ptregs(&u, task_pt_regs(target));
		/*
		 * A guest's bases come from the vmctx-owned copy, which a
		 * context switch cannot clobber; thread.fsbase answers only
		 * for a task nothing has SETREGS'd. See guest_fsbase in the
		 * struct for the measured failure.
		 */
		if (vc->fsgs_valid) {
			u.fs_base = vc->guest_fsbase;
			u.gs_base = vc->guest_gsbase;
		} else {
			u.fs_base = target->thread.fsbase;
			u.gs_base = target->thread.gsbase;
		}
		if (copy_to_user(arg, &u, sizeof(u)))
			ret = -EFAULT;
		break;
	}
	case VMCTX_CTL_SETREGS: {
		struct vmctx_uregs u;

		if (copy_from_user(&u, arg, sizeof(u))) {
			ret = -EFAULT;
			break;
		}
		ptregs_from_uregs(task_pt_regs(target), &u);
		/*
		 * The thread pointer travels with the registers -- into the
		 * vmctx-owned copy first, which is what a guest's entry path
		 * reads and what no context switch can undo. thread.fsbase is
		 * still written for the service-context half, which has no
		 * backend to read the copy; zero still means "leave it alone".
		 */
		if (u.fs_base) {
			vc->guest_fsbase = u.fs_base;
			vc->fsgs_valid = 1;
			x86_fsbase_write_task(target, u.fs_base);
		}
		if (u.gs_base) {
			vc->guest_gsbase = u.gs_base;
			vc->fsgs_valid = 1;
			x86_gsbase_write_task(target, u.gs_base);
		}
		break;
	}
	case VMCTX_CTL_PEEK:
		ret = vmctx_ctl_mem(target, arg, false);
		break;
	case VMCTX_CTL_POKE:
		ret = vmctx_ctl_mem(target, arg, true);
		break;
	case VMCTX_CTL_TAKE:
		ret = vmctx_ctl_take(target, arg);
		break;
	case VMCTX_CTL_TAKEOBJ:
		ret = vmctx_ctl_takeobj(target, arg);
		break;
	case VMCTX_CTL_PROTECT:
		ret = vmctx_ctl_protect(target, arg);
		break;
	case VMCTX_CTL_PROTECTOBJ:
		ret = vmctx_ctl_protectobj(target, arg);
		break;
	case VMCTX_CTL_MAPOBJ:
		ret = vmctx_ctl_mapobj(target, arg);
		break;
	case VMCTX_CTL_APPLYMAP:
		ret = vmctx_ctl_applymap(target, arg);
		break;
	case VMCTX_CTL_SYSCALL: {
		struct vmctx_syscall rq;
		long left;

		if (copy_from_user(&rq, arg, sizeof(rq))) {
			ret = -EFAULT;
			break;
		}
		if (vc->sys_pending) {	/* one at a time per context */
			ret = -EBUSY;
			break;
		}
		vc->sys_nr   = rq.nr;
		memcpy(vc->sys_args, rq.args, sizeof(vc->sys_args));
		vc->sys_ret     = 0;
		vc->sys_done    = 0;
		vc->sys_pending = 1;
		/*
		 * Nudge it: a context parked waiting for its monitor is asleep
		 * on reply_wq, and it has to reach the return-to-user path
		 * before it can run anything for us.
		 */
		wake_up(&vc->reply_wq);
		wake_up_process(target);
		/*
		 * Uninterruptibly, and deliberately. Giving up here would leave
		 * the request pending and the context would perform it later,
		 * with nobody to take the answer -- and a syscall performed
		 * twice, or performed after the caller has stopped caring, is
		 * the class of bug this file has been bitten by before.
		 */
		left = wait_event_timeout(vc->sys_wq,
					  vc->sys_done || vc->dead, 30 * HZ);
		if (!vc->sys_done) {
			vc->sys_pending = 0;
			pr_warn("vmctx: pid %d: did not perform the monitor's syscall %llu%s\n",
				task_pid_nr(target), rq.nr,
				left == 0 ? " (timed out)" : " (context gone)");
			ret = -ETIMEDOUT;
			break;
		}
		rq.ret     = vc->sys_ret;
		rq.changed = vc->sys_changed;
		rq.regs    = vc->sys_regs;
		ret = copy_to_user(arg, &rq, sizeof(rq)) ? -EFAULT : 0;
		break;
	}
	case VMCTX_CTL_GET_CLEAR_TID: {
		/*
		 * The clear_child_tid the real clone recorded. Handed to the
		 * monitor so it can clear that word coherently at thread exit --
		 * take the page, write zero, wake a joiner -- instead of do_exit
		 * writing it into a page the machine running the guest may hold,
		 * which the guest never sees and pthread_join then spins on.
		 */
		u64 word = (u64)(unsigned long)target->clear_child_tid;

		ret = copy_to_user(arg, &word, sizeof(word)) ? -EFAULT : 0;
		break;
	}
	default:
		ret = -EINVAL;
	}

out_vc:
	kref_put(&vc->kref, vmctx_task_free);
out_put:
	put_task_struct(target);
	return ret;
}

SYSCALL_DEFINE3(vmctx_ctl, pid_t, pid, unsigned int, cmd, void __user *, arg)
{
	return vmctx_ctl_impl(pid, cmd, arg);
}

SYSCALL_DEFINE1(vmctx_run, struct vmctx_run_config __user *, ucfg)
{
	return vmctx_run_current(ucfg);
}
