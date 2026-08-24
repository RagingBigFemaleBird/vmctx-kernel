/* SPDX-License-Identifier: GPL-2.0 */
/*
 * vmctx — native "VM context" for Linux (avm.md project).
 *
 * Kernel-internal interface. The core (kernel/vmctx.c) owns the task
 * lifecycle, the scheduler/signal integration, and the vmctx_run(2)
 * syscall. The hardware backend (Intel VMX / AMD SVM), which actually
 * enters guest mode, is provided by a module that registers a
 * struct vmctx_backend. Keeping the backend separate lets the risky
 * guest-entry code iterate as a module while the native integration lives
 * in the kernel image.
 */
#ifndef _LINUX_VMCTX_H
#define _LINUX_VMCTX_H

#include <linux/types.h>
#include <linux/atomic.h>
#include <linux/wait.h>
#include <linux/kref.h>
#include <linux/mm_types.h>
#include <uapi/linux/vmctx.h>

/*
 * Kernel-internal reply action: nobody serviced the event, but the monitor is
 * fine — re-take it after the signal that interrupted the wait is delivered.
 * Not part of the ABI: a monitor never sends this.
 */
#define VMCTX_ACT_RETRY 0xffff

struct task_struct;
struct module;
struct pt_regs;
struct vmctx_backend;
struct vm_fault;

/*
 * Per-task VM-context state. Pointed to by task_struct.vmctx (NULL for
 * ordinary tasks). Allocated by the core when a task enters vmctx_run and
 * freed when it returns. backend_priv holds the backend's guest state
 * (VMCB/NPT for SVM).
 */
struct vmctx_task {
	/*
	 * Refcounted: task_struct.vmctx holds one reference, and a monitor
	 * inside vmctx_ctl(2) holds one for the duration of the call. Without
	 * this a monitor sleeping in VMCTX_CTL_WAIT would be parked on a wait
	 * queue inside memory the context frees when it exits.
	 */
	struct kref kref;
	int    dead;			/* context has been torn down          */
	void  *backend_priv;
	struct vmctx_backend *backend;	/* backend that owns this guest      */
	void __user *ucfg;		/* user vmctx_run_config for writeback */
	int    active;			/* 1 while the guest should keep running */
	/*
	 * The address space this context counted itself into (see
	 * vmctx_mm_note). Remembered rather than re-read from t->mm at exit,
	 * because execve replaces t->mm and a decrement has to land on the
	 * same slot the increment did or the count is nonsense.
	 */
	struct mm_struct *noted_mm;
	int    released;		/* monitor said "go" (see WAIT_MONITOR)  */
	int    needs_rebuild;		/* execve replaced the address space    */
	int    exit_process;		/* guest exit ends the whole process    */
	__u64  flags;			/* VMCTX_FLAG_* from vmctx_run         */
	__u64  entry;			/* USERCODE guest entry RIP           */
	__u64  stack;			/* USERCODE guest RSP                 */
	__s32  backing_fd;		/* shared object faults are backed from */
	/*
	 * The object shared with every context of the same program, for ranges
	 * the owner says are shared. A fork copies backing_fd; this it must
	 * not, which is the whole reason it is separate.
	 */
	__s32  shared_fd;
	/*
	 * Set while a fault is being reported from the kernel's own page-fault
	 * path, where the caller already holds mmap_read_lock and may hold a
	 * page-table lock. Anything that inspects the address space has to know
	 * that, because taking either again is a deadlock and not a slow path.
	 */
	int    in_mm_fault;
	/*
	 * VMCTX_PGC_* for the fault being reported from the mm hook: the class
	 * of the mapping it was taken in, read from the VMA while it is in
	 * hand. Only vmctx_anon_fault() sets it, only for the duration of the
	 * report, and vmctx_redirect_fault() copies it into the event. Zero
	 * the rest of the time, so a backend-reported fault carries no class.
	 */
	__u32  mm_fault_class;
	__u64  enters;			/* VMRUN count                 */
	__u64  exits;			/* VMEXIT count                */
	__u64  counter;			/* guest-observable counter    */
	__u64  faults;			/* guest page faults serviced  */
	__u64  max_exits;		/* 0 = unlimited               */
	__u32  last_exit_reason;
	__s32  exit_status;

	/*
	 * The GUEST's FS/GS bases, owned by vmctx rather than borrowed from
	 * t->thread. thread.fsbase cannot carry a guest's base safely: the
	 * backend's entry refresh used to read the LIVE MSR, and a SETREGS to
	 * a task still on a CPU never reaches that register -- the guest then
	 * entered with the MONITOR's own thread pointer while thread.fsbase
	 * read back correct. SETREGS stores here, the backend loads the
	 * VMCB/VMCS from here on every entry and saves back on every exit,
	 * and GETREGS answers from here. See the 7.0.14 twin for the measured
	 * failure (ws1 threads dying in __ctype_init on another program's TLS).
	 */
	__u64  guest_fsbase;
	__u64  guest_gsbase;
	int    fsgs_valid;

	/*
	 * Redirection to a monitor process (avm.md objectives 2/3). When a
	 * monitor is attached and the event class is redirected, the context
	 * publishes the event, wakes the monitor, and sleeps until it replies.
	 */
	struct task_struct *monitor;
	struct vmctx_event ev;
	struct vmctx_reply reply;
	int    ev_pending;
	/*
	 * Serialises publishing an event, the monitor dequeuing it, and the
	 * waiter giving up on it -- the three transitions that decide whether
	 * a syscall runs once or twice.
	 *
	 * Without it those were plain loads and stores, and the gap between
	 * the waiter reading ev_taken and clearing ev_pending was wide enough
	 * for the monitor to take the event in between. The waiter then
	 * rewound the instruction and the guest re-executed a syscall the
	 * monitor was already running: th9 emitted forty lines and printed
	 * forty-one, the same "thread 36 ok" twice, about once in fifty runs.
	 * Discarding the late reply -- which is what ev_abandoned does -- does
	 * not undo the write(2) that has already happened.
	 */
	spinlock_t ev_lock;
	/*
	 * A reported event whose waiter gave up because a signal arrived. The
	 * monitor may answer it anyway, and that answer must be discarded
	 * rather than applied to the next event.
	 *
	 * Under ev_lock a waiter can only give up on an event the monitor has
	 * not taken, so this should now never be set. It is kept because it is
	 * the check that says so.
	 */
	int    ev_abandoned;
	/*
	 * Whether the monitor has actually dequeued the event now pending. An
	 * event the monitor never saw needs no discarding — marking it
	 * abandoned anyway means the reply to the *next* event is thrown away
	 * instead, and then nobody is waiting for anything.
	 */
	int    ev_taken;
	/* The monitor died or was dropped, as opposed to detaching on purpose. */
	int    monitor_lost;
	/*
	 * Set while the backend is deliberately letting this kernel service a
	 * fault the monitor already declined. Without it that fault would be
	 * reported a second time from the page-fault path below.
	 */
	int    in_local_service;
	int    reply_pending;
	/*
	 * A syscall the monitor asked this context to make on its own behalf.
	 *
	 * Requested with VMCTX_CTL_SYSCALL and performed here, in the context's
	 * own task, because that is the only place the arguments mean what the
	 * program meant. The monitor sleeps on sys_wq until sys_done.
	 */
	int    sys_pending;
	int    sys_done;
	__u64  sys_nr;
	__u64  sys_args[6];
	__s64  sys_ret;
	__u32  sys_changed;		/* VMCTX_REG_* the call wrote        */
	struct vmctx_uregs sys_regs;	/* the register set after the call   */
	/*
	 * Pages the in-flight assisted syscall has faulted in, held against
	 * hand-over until the call completes.
	 *
	 * A syscall that faults a page in is going to touch it -- that is what
	 * the fault was -- and for a read(2) with O_DIRECT the touch is a
	 * device writing the page after get_user_pages() pins it. Between the
	 * fault-in and the pin there is a window in which the page's reference
	 * count is exact and a VMCTX_CTL_TAKE claim rightly succeeds, and
	 * tests/dma1.c measured what happens then: the other machine is handed
	 * the page's pre-DMA bytes, the device completes into a page nobody
	 * calls current any more, and the guest compares zeros against the
	 * file. 4 to 182 bad rounds in 200, first at round 0.
	 *
	 * So the fault path records the address here while sys_pending's call
	 * is in flight, and the take paths answer -EBUSY for a recorded page
	 * until the call ends. -EBUSY already means "wait and ask again" to
	 * every caller. The list is cleared when the call completes; if it
	 * overflows, sys_pages_over holds *every* page until the call ends,
	 * and says so in the log -- a silent cap here would be a silent
	 * corruption window.
	 *
	 * This is the escalation lock, expressed where the facts are: the
	 * syscall is the atom, the page is the unit, and the wait is bounded
	 * by the syscall's own duration rather than by a guess.
	 */
	/*
	 * Each entry expires VMCTX_HOLD_TTL after it was recorded, and that
	 * expiry is load-bearing, not a hedge. The hold's one job is the
	 * fault-to-pin window: after get_user_pages() pins the page, the
	 * claim's own pin check refuses hand-over without any help from this
	 * list. A hold that instead lasted the whole syscall deadlocked with
	 * the syscall's own semantics -- measured on futex_wait, whose entry
	 * write-faults the caller's stack page (get_futex_key does a write
	 * GUP), which recorded the stack in this list, which stopped the
	 * other machine's guest -- who needs that stack to run -- from ever
	 * issuing the FUTEX_WAKE the sleep was waiting for. Group kernel
	 * stacks at the failure: the holder asleep in futex_do_wait inside
	 * vmctx_dispatch_syscall, its sibling parked idle.
	 */
	spinlock_t sys_pages_lock;
	int    in_monitor_syscall;
	int    sys_npages;
	int    sys_pages_over;
	__u64  sys_over_when;
	__u64  sys_pages[64];
	__u64  sys_pages_when[64];
	wait_queue_head_t sys_wq;
	wait_queue_head_t ev_wq;	/* monitor waits here for events     */
	wait_queue_head_t reply_wq;	/* context waits here for the reply  */

	/*
	 * The guest's secondary TLB, and the invalidations vmctx does not
	 * perform itself.
	 *
	 * backend->invalidate is called from the four places vmctx takes a page
	 * away on purpose -- take, takeobj, protect, protectobj. Everything ELSE
	 * that moves a folio out from under a running guest goes unnoticed:
	 * reclaim, migration, compaction, THP collapse, and the guest's own
	 * munmap/mprotect. Each of those clears a PTE the guest may still be
	 * holding in its ASID-tagged TLB, which no host flush reaches.
	 *
	 * An mmu_notifier is the kernel's own answer to exactly this, and it is
	 * what KVM uses. Registered per context, on the mm the guest runs on,
	 * so those paths reach the same shootdown the deliberate ones do.
	 *
	 * void * rather than an embedded struct mmu_notifier: keeping
	 * <linux/mmu_notifier.h> out of this header, which is included from the
	 * scheduler's own. It points at a struct vmctx_mn in kernel/vmctx.c.
	 */
	void  *mn;
};

/*
 * Dispatch one guest syscall through the host's generic syscall table.
 *
 * @regs must be the VM-context task's own pt_regs, holding the guest's
 * register state with the syscall number in orig_ax. The backend keeps that
 * frame in sync with its hardware save area (VMCB), which is what makes the
 * kernel's existing machinery — argument fetch, signal delivery, syscall
 * restart, rt_sigreturn's full register restore, ptrace — operate on guest
 * state. Nothing here is syscall-specific: any syscall the kernel implements
 * is reachable, so backends never need per-syscall code.
 *
 * Returns the syscall's return value (also stored in regs->ax), or -ENOSYS for
 * an out-of-range number or one on the small deny list. Must be called from
 * the task's own context with interrupts enabled and no locks held — i.e. from
 * the return-to-user path, where syscall context is valid.
 */
long vmctx_dispatch_syscall(struct pt_regs *regs);

/* backend->run() return values */
#define VMCTX_RUN_CONTINUE 0	/* guest yielded; keep running          */
#define VMCTX_RUN_EXIT     1	/* guest requested exit (see exit_status) */

struct vmctx_backend {
	struct module *owner;
	/* Build per-task guest state (VMCB/NPT). Store it in t->vmctx->backend_priv. */
	int  (*create)(struct task_struct *t);
	/*
	 * fork(): duplicate src's guest state into dst, which has its own
	 * (copy-on-write) address space. The child's guest must resume at the
	 * same instruction with a 0 return value, like fork() in user mode.
	 */
	int  (*clone)(struct task_struct *dst, struct task_struct *src);
	/* Run the guest until one VMEXIT that needs core attention.
	 * Returns VMCTX_RUN_CONTINUE, VMCTX_RUN_EXIT, or -errno. */
	int  (*run)(struct task_struct *t);
	/* Tear down per-task guest state. */
	void (*destroy)(struct task_struct *t);
	/*
	 * The host has just removed or write-protected a page the guest was
	 * running on (a take, an object-wide take, or a write-protect). The
	 * guest may be executing this instant on other CPUs with the old
	 * translation cached in its own hardware TLB, which is tagged by a guest
	 * ASID the host's TLB flush never reaches. Force any such guest out of
	 * guest mode so its next entry reloads the mapping (an entry flushes the
	 * guest ASID). Called after the PTE change and its host-side flush, while
	 * the folio is still pinned, so no store through a stale mapping can
	 * outlive this. addr is the page; a backend may flush wholesale and
	 * ignore it. Optional -- NULL if the backend keeps no separate TLB.
	 */
	void (*invalidate)(struct task_struct *t, unsigned long addr);
};

/*
 * Said out loud so the out-of-tree module can ask, because it is built against
 * two kernels that do not both have the hook above: 7.0.14 here, and 6.18.35 on
 * the Intel netboot target, which is held at that version on purpose. A struct
 * member cannot be tested by the preprocessor, and sniffing LINUX_VERSION_CODE
 * would answer a question about a version rather than about a feature -- and
 * answer it wrongly the day 6.18.35 gains the hook.
 */
#define VMCTX_BACKEND_HAS_INVALIDATE 1

int  vmctx_register_backend(struct vmctx_backend *b);
void vmctx_unregister_backend(struct vmctx_backend *b);

/* VMEXIT-scan lost-write detector: the backend calls this after each VMRUN so
 * the guest's own stores seed the high-water-mark. No-op unless vmctx_watch_gva
 * is set. */
void vmctx_watch_sample(void);

/* The watched guest VA (0 if unarmed), so the backend can cheaply skip the
 * post-serve read below for every other page. */
unsigned long vmctx_watch_page(void);

/* Fault-serve counterpart of vmctx_watch_check: the backend calls this with the
 * eight slot words the guest is about to READ BACK, right after a fault has been
 * served and before the instruction retries. Flags a read below the mark seeded
 * at take time -- the stale read the VMEXIT sampler cannot see. sl8 points to 8
 * unsigned longs read from guest VA (page base + 64). No-op unless armed. */
void vmctx_watch_served(unsigned long gva, const unsigned long *sl8);

/*
 * The three above are a DIAGNOSTIC, not part of the model, and 6.18.35 does not
 * carry them. See VMCTX_BACKEND_HAS_INVALIDATE for why this is a feature macro
 * and not a version test. Worth keeping off the other kernel on its own merits:
 * the VMEXIT-scan reading these produced was retracted -- a two-read compare
 * against a monotonic writer, which manufactured the "guest behind the object"
 * result it reported -- and vmctx_watch_sample() perturbs enough that it sits
 * behind a module parameter defaulting to off.
 */
#define VMCTX_HAS_WATCH 1

/* Implementation of the vmctx_run(2) syscall body. */
long vmctx_run_current(struct vmctx_run_config __user *ucfg);

/*
 * Native switch-path integration. The return-to-user loop
 * (exit_to_user_mode_loop) runs the guest for a VM-context task: it enters
 * the guest at switch-in and, when a guest VMEXIT sets need_resched, the
 * loop's own schedule() performs the context switch-out. The active count
 * keeps the check free for ordinary tasks.
 */
extern atomic_t vmctx_nr_active;
bool vmctx_task_active(struct task_struct *t);
/*
 * Whether the return-to-user path must hold a pending signal back rather than
 * deliver it. True only for a service context between forwarded syscalls; see
 * the definition. The atomic read in the wrapper keeps this free for every
 * other task on the machine.
 */
bool vmctx_defer_signal_work(struct task_struct *t, unsigned long ti_work);
static inline bool vmctx_defer_signals(struct task_struct *t,
				       unsigned long ti_work)
{
	return atomic_read(&vmctx_nr_active) &&
	       vmctx_defer_signal_work(t, ti_work);
}
void vmctx_guest_step(struct pt_regs *regs);
void vmctx_task_exit(struct task_struct *t);

/*
 * Process-lifecycle integration, so the usual primitives work on a VM
 * context:
 *   vmctx_copy_task()   fork/clone — give the child its own copy of the VM
 *                       context (never the parent's pointer) so it continues
 *                       as a guest in its own address space.
 *   vmctx_exec_notify() execve — the address space the guest was running on
 *                       is gone; mark the context for rebuild so the guest is
 *                       re-entered at the new program's entry point, i.e. the
 *                       exec'd program runs inside the VM context.
 */
int  vmctx_copy_task(struct task_struct *dst, struct task_struct *src,
		     unsigned long clone_flags);
void vmctx_exec_notify(struct task_struct *t);

/*
 * Redirection hooks used by a backend. Each returns:
 *   0  — the kernel should handle the event itself (no monitor, event class not
 *        redirected, or the monitor asked for VMCTX_ACT_SELF);
 *   1  — the monitor handled it (for a syscall, regs->ax already holds the
 *        value to give the guest); the backend should just resume;
 *  <0  — the context should be terminated with this status.
 */
/* mm/mprotect.c: apply a protection to the calling context's own mm. */
long vmctx_mprotect(unsigned long start, size_t len, unsigned long prot);

int vmctx_redirect_syscall(struct task_struct *t, struct pt_regs *regs);
int vmctx_redirect_fault(struct task_struct *t, struct pt_regs *regs,
			 __u64 vec, __u64 addr, __u64 err);

/*
 * Asked before this kernel supplies a page for a task from anything other than
 * the one place that can be right.
 *
 * Called from two places: do_anonymous_page(), where a zero page would be
 * invented, and do_fault() for a *writable private* file mapping, where the
 * file's contents would be used for a range whose contents the program has
 * since changed on the machine that runs it. Both are the same question --
 * whose memory is this? -- and the answer is never this kernel's to make up.
 *
 * A VM context's memory is not this kernel's to invent — it belongs wherever
 * the program came from, and the monitor is the only thing that can supply it.
 * Faults taken *in guest mode* reach the monitor through the backend, but a
 * context takes faults in other ways too: this kernel reading a syscall
 * argument with copy_from_user(), ptrace, process_vm_readv. Those are ordinary
 * faults, and answering them with a zero page fabricates memory — silently, and
 * permanently, because the page is present afterwards so nothing faults there
 * again. The program reads zeros for the rest of its life, arbitrarily far from
 * whatever caused it.
 *
 * One hook, so every way of touching a context's memory is supplied the same
 * way, and no caller has to know which syscall it is in.
 *
 * Returns 0 if the monitor supplied it (retry the access), or VM_FAULT_FALLBACK
 * to let the normal anonymous path run — which is the right answer for a
 * context's own memory, e.g. its stack.
 */
vm_fault_t vmctx_anon_fault(struct vm_fault *vmf);

/*
 * An anonymous fault on an address space that is not the faulting task's.
 *
 * This is how a monitor's own reads and writes of a context's memory reach the
 * fault path: get_user_pages_remote() faults the context's mm while current is
 * the monitor. vmctx_anon_fault()'s hook asks current->vmctx and therefore does
 * not fire, so the page is supplied by do_anonymous_page() with nobody asked.
 * Counted here so that the size of that hole is a number rather than a
 * deduction; /sys/module/kernel/parameters/vmctx_foreign_*.
 */
vm_fault_t vmctx_foreign_anon_fault(struct mm_struct *mm, int write);

/*
 * The same question for a FILE fault, which is the one with no hook at all.
 *
 * On a destination every guest page is a MAP_SHARED mapping of the context's
 * backing memfd, so a monitor reading a context's memory faults a shared file
 * VMA of somebody else's mm -- past vmctx_foreign_anon_fault() (anon only) and
 * past do_fault()'s own hook (current->vmctx, and !VM_SHARED). shmem_fault()
 * then inserts a fresh zero folio INTO THE OBJECT, which is the address
 * space's memory and not one task's page table.
 * /sys/module/kernel/parameters/vmctx_ffile_*.
 */
/*
 * Every fault on a live context's memory, counted where they all pass through,
 * against the ones a monitor was actually asked about. The difference is
 * "faults on a context's pages that nobody was asked about", and counting it at
 * handle_mm_fault() rather than per path is the point: a per-path audit is how
 * a path gets overlooked. /sys/module/kernel/parameters/vmctx_f_*.
 */
void vmctx_fault_seen(struct vm_area_struct *vma, unsigned int flags);
void vmctx_fault_reported(void);

vm_fault_t vmctx_foreign_file_fault(struct mm_struct *mm, int write,
				    int shared, struct address_space *mapping,
				    pgoff_t pgoff);
/*
 * mm/shmem.c reports every folio it CREATES (shmem_get_folio_gfp's allocation)
 * so the ones made in a live context's backing object are counted by path --
 * the context's own user or kernel-mode fault, a foreign read or write through
 * GUP, a write(2)/fallocate on the object -- and stamped per page, so TAKEOBJ
 * can name the maker of an all-zero page it captures. Gated on
 * vmctx_nr_active by the caller. /sys/module/kernel/parameters/vmctx_shm_*.
 */
void vmctx_shmem_alloced(struct address_space *mapping, struct folio *folio,
			 struct vm_fault *vmf, int sgp_write, int sgp_falloc);
/*
 * ...and asked BEFORE it allocates: a foreign READ fault on a hole in a
 * context's object is refused (false -> -EFAULT, a short read for the
 * monitor) rather than answered with a zero folio that becomes the address
 * space's memory. The door session 36 named; see the definition.
 */
bool vmctx_shmem_may_alloc(struct address_space *mapping, pgoff_t index,
			   struct vm_fault *vmf);

/*
 * A write to a page of a context that is present but not writable -- the third
 * way into a context's memory, and the only one that cannot ask the monitor:
 * do_wp_page() holds the page-table lock. Counted only, split by who faulted
 * and by what kind of mapping, so the shape of the hole is a number before
 * anything is built on it. /sys/module/kernel/parameters/vmctx_wp_*.
 */
/*
 * Whether a fault in this address space must install exactly one page.
 *
 * A context's memory is coherent page by page: pages change machines one at a
 * time, and the ownership record on the owning side is kept per page. The
 * fault paths' batch installs -- a large page-cache folio mapped whole by
 * finish_fault(), a PMD-sized text mapping, a multi-page anonymous folio --
 * install neighbours nobody faulted on, and for a context a neighbour may be
 * a page whose current bytes are on the other machine. A batch install puts
 * this side's stale copy over it, present, so it never faults again and no
 * hook can ever ask. One page per fault is the repair: the neighbours stay
 * absent, and each one is asked about when it is actually touched.
 *
 * Both faulters matter: the context itself (current->vmctx), and anything
 * reaching a context's mm from outside -- a monitor's PEEK/POKE through GUP
 * faults with current being the monitor. The mm test catches those.
 */
bool vmctx_mm_is_context(struct mm_struct *mm);
bool vmctx_install_single(struct mm_struct *mm);
extern bool vmctx_fault_ro_file;

vm_fault_t vmctx_wp_fault(struct mm_struct *mm, int own, int shared, int filed,
			  int zero);

/* Implementation of the vmctx_ctl(2) syscall body. */
long vmctx_ctl_impl(pid_t pid, unsigned int cmd, void __user *arg);

static inline bool vmctx_should_run(struct task_struct *t)
{
	return atomic_read(&vmctx_nr_active) && vmctx_task_active(t);
}

#endif /* _LINUX_VMCTX_H */
