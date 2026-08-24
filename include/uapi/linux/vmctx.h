/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * vmctx — native "VM context" for Linux (avm.md project).
 *
 * ABI for the vmctx_run(2) syscall. A process that calls vmctx_run becomes
 * a VM context: it remains an ordinary task (pid, /proc, signals,
 * scheduling) but its execution is carried by a hardware guest (VMRUN)
 * driven by the registered backend. The syscall returns when the guest
 * exits, an error occurs, max_exits is reached, or a signal arrives.
 */
#ifndef _UAPI_LINUX_VMCTX_H
#define _UAPI_LINUX_VMCTX_H

#include <linux/types.h>

/* vmctx_run_config.flags */
#define VMCTX_FLAG_USERCODE	(1u << 0)  /* run guest at .entry/.stack in the
					    * caller's address space, forwarding
					    * its syscalls to the host (Dune model).
					    * Without this: the built-in demo
					    * counter guest. */
#define VMCTX_FLAG_EXIT_PROCESS	(1u << 1)  /* guest exit()/exit_group() exits the
					    * whole process with that status,
					    * instead of returning from
					    * vmctx_run(2). Required when the
					    * guest execs a program: there is no
					    * meaningful frame to return to. */

#define VMCTX_FLAG_REDIRECT_SYSCALL (1u << 2) /* guest syscalls are reported to a
					       * monitor process instead of being
					       * handled by the kernel */
#define VMCTX_FLAG_REDIRECT_FAULT   (1u << 3) /* likewise for guest faults    */
#define VMCTX_FLAG_WAIT_MONITOR     (1u << 4) /* vmctx_run(2) blocks until a
					       * monitor attaches before the
					       * guest starts. Without this a
					       * monitor cannot reliably attach:
					       * the context must exist first,
					       * but a short-lived guest can
					       * finish before anyone sees it. */
/*
 * A context that holds an address space and performs syscalls in it, and never
 * executes anything.
 *
 * This is what the machine that *owns* a program needs to be. It has the
 * program's memory, its descriptors, its credentials and its identity, and it
 * answers VMCTX_CTL_SYSCALL -- but the program's instructions run somewhere
 * else entirely, so there is nothing here to enter.
 *
 * It follows that a service context needs no virtualisation hardware: nothing
 * is ever entered, so no backend is required and none is created. A machine
 * with no VMX and no SVM can still own a program.
 */
#define VMCTX_FLAG_SERVICE          (1u << 6)

#define VMCTX_FLAG_RESTORE          (1u << 5) /* start the guest from the task's
					       * pt_regs instead of .entry/.stack,
					       * so a monitor can SETREGS a
					       * checkpointed register set and have
					       * the guest resume mid-execution.
					       * Use with WAIT_MONITOR. */

struct vmctx_run_config {
	__u64 flags;		/* VMCTX_FLAG_*                              */
	__u64 max_exits;	/* 0 = run until killed; else stop after N   */
	__u64 entry;		/* USERCODE: guest entry RIP (user address)  */
	__u64 stack;		/* USERCODE: guest RSP (user address)        */
	/*
	 * Where this context's memory is backed from, and how contexts come to
	 * share memory.
	 *
	 * A descriptor for a shared object -- a memfd -- in which a guest virtual
	 * address is its own offset. A fault at A is backed by mapping this object
	 * at A with offset A, so every context started with the *same* object sees
	 * the same page at the same address, and a write by one is seen by the
	 * others. That is the whole of "sharing an address space": a thread is a
	 * context handed its parent's object, and a process is a context handed a
	 * new one, which is the snapshot a fork wants.
	 *
	 * The destination is told nothing else about memory -- no ranges, no
	 * layout, no mapping decisions. It faults, asks the source for the bytes,
	 * and puts them in the object. -1 means anonymous private backing, which
	 * shares with nobody.
	 */
	__s32 backing_fd;
	/*
	 * A second object, shared with every context of the same program.
	 *
	 * backing_fd is this context's own memory: a thread is handed its
	 * parent's, a process a copy of it, and that copy is what fork means.
	 * Shared memory is the one thing a fork must *not* copy, and an offset
	 * cannot express it -- two offsets in two different objects are two
	 * pages however they are numbered. So a range the owner calls shared is
	 * mapped from this one instead, which parent and child hold alike.
	 *
	 * -1 for a context that shares nothing.
	 */
	__s32 shared_fd;
	/* outputs, filled in by the kernel before returning: */
	__u64 out_enters;	/* number of VMRUNs performed                */
	__u64 out_exits;	/* number of VMEXITs observed                */
	__u64 out_counter;	/* guest-maintained counter / syscalls served */
	__u64 out_faults;	/* guest page faults serviced by the host    */
	__u32 out_exit_reason;	/* last VMEXIT reason                        */
	__s32 out_status;	/* guest exit status                         */
};

/*
 * ---------------------------------------------------------------------------
 * Fault/syscall redirection (avm.md objectives 2 and 3).
 *
 * A VM context can either handle its own faults and syscalls (the default —
 * what a context hosting a plain user program wants) or have them redirected
 * to another process, a "monitor" (what a context hosting a VM wants). The
 * monitor drives the context with vmctx_ctl(2): it waits for events, inspects
 * or edits the context's registers and memory, and resumes it.
 *
 * This is also the mechanism remote execution is built on: a monitor is free to
 * service an event however it likes, including forwarding it over a network to
 * another machine.
 * ---------------------------------------------------------------------------
 */

/* vmctx_ctl(2) commands */
#define VMCTX_CTL_ATTACH	1  /* become the monitor for <pid>            */
#define VMCTX_CTL_DETACH	2  /* stop monitoring; context self-handles   */
#define VMCTX_CTL_WAIT		3  /* block for the next event (vmctx_event)  */
#define VMCTX_CTL_RESUME	4  /* answer the event (vmctx_reply)          */
#define VMCTX_CTL_GETREGS	5  /* read guest registers (vmctx_uregs)      */
#define VMCTX_CTL_SETREGS	6  /* write guest registers (vmctx_uregs)     */
#define VMCTX_CTL_PEEK		7  /* read context memory (vmctx_mem)         */
#define VMCTX_CTL_POKE		8  /* write context memory (vmctx_mem)        */
#define VMCTX_CTL_TAKE		9  /* remove memory from the context and return
				    * its last contents (vmctx_mem); 0 means
				    * the page was not there to take           */
/*
 * Make the context execute a syscall in its OWN address space.
 *
 * The monitor supplies the number and the arguments; the context runs it. That
 * distinction is the whole point and it is not a convenience. A monitor is a
 * different process with a different address space, so any argument that is a
 * pointer means something else to it than it does to the program -- the same
 * address is different memory. A monitor that runs the call itself is therefore
 * answering with its own memory and cannot be right in general; the only way to
 * be right is for the task that owns the address space to make the call.
 *
 * That is also what makes descriptors, credentials and the process identity
 * come out right, since they are the context's own rather than the monitor's.
 *
 * Nothing here is syscall-specific: the number is dispatched through the
 * kernel's own table, so anything the kernel implements is reachable.
 */
#define VMCTX_CTL_SYSCALL	11
/*
 * Make a task that already holds a program into a service context.
 *
 * The program has to be loaded by the kernel that owns it -- an execve, with
 * its interpreter, its libraries and its initial stack -- and only then is
 * there an address space worth holding. The task cannot ask for this itself:
 * by the time it is interesting it *is* the program, and the program knows
 * nothing about any of this. So the monitor asks on its behalf, and from that
 * point the task never executes another instruction of its own.
 */
#define VMCTX_CTL_ADOPT		12
/*
 * Hand a page over from the whole address space at once: unmap it from every
 * context that has it, copy it, and remove it from the backing object, with no
 * moment in between where any of them can write it. VMCTX_CTL_TAKE does this
 * for one context and leaves the page in the object, so several of them in a
 * row is a sequence with a gap at every join. -ENOENT means the range is not
 * object-backed and the caller should fall back to VMCTX_CTL_TAKE.
 */
#define VMCTX_CTL_TAKEOBJ	13
/*
 * Read back the context's clear_child_tid -- the word CLONE_CHILD_CLEARTID
 * named when the source ran the guest's real clone. The monitor uses it to
 * clear that word coherently at thread exit (take the page, write zero, wake a
 * joiner) rather than let do_exit write it into a page the guest may hold. Only
 * the kernel that ran the clone has the value; this hands it back so no one has
 * to parse clone_args. arg is a __u64*; the pointer value (a guest address) is
 * written to it, or 0 if the context named none.
 */
#define VMCTX_CTL_GET_CLEAR_TID	14
/*
 * Write-protect a page across the WHOLE address space at once -- every context
 * mapping the object folio, atomically, under the folio lock. The read-claim
 * counterpart of VMCTX_CTL_TAKEOBJ: after it returns, no context can write the
 * page until it faults, so the copy it hands back cannot be one iteration stale.
 * VMCTX_CTL_PROTECT does one context and leaves siblings writable, which is a
 * window a sibling can write through -- the lost-write both-hold race. -ENOENT
 * means the range is not object-backed; 0 means the object does not have it.
 */
#define VMCTX_CTL_PROTECTOBJ	15
/*
 * Give a context the page its OWN backing object already holds, by MAPPING it
 * rather than copying it.
 *
 * Every context of one address space maps the same backing object MAP_SHARED at
 * offset == the guest address, so for a given address they all map ONE folio --
 * that is what memory sharing means here, and a second physical page for the
 * same address would break it. When a page is in the object and a context has
 * not faulted it in yet, the bytes are therefore already in the right physical
 * page and nothing needs to move: this faults the page into that context so its
 * page table points at that same folio.
 *
 * The alternative the monitor had was to read the object into a buffer and poke
 * it back through the context, which is a read-then-write-back of the SAME folio
 * -- it copies a page onto itself and destroys any store a sibling context lands
 * between the read and the write. This exists so that path is never taken for a
 * page the address space already holds.
 */
#define VMCTX_CTL_MAPOBJ	16
/*
 * Apply a mapping change (a struct vmctx_reply's map_op/addr/len/off/prot) to a
 * context that did not make the call. A guest's threads share one address space;
 * here each context has its own mm, so a reshape by one has to be given to all.
 */
#define VMCTX_CTL_APPLYMAP	17
#define VMCTX_CTL_PROTECT	10 /* take write permission away from a page,
				    * leaving it readable (vmctx_mem); a write
				    * then arrives as a protection fault       */

/* event types */
#define VMCTX_EV_NONE		0
#define VMCTX_EV_SYSCALL	1
#define VMCTX_EV_FAULT		2

struct vmctx_event {
	__u32 type;		/* VMCTX_EV_*                               */
	__s32 pid;		/* the context's pid                        */
	__u64 nr;		/* SYSCALL: number; FAULT: exception vector  */
	/*
	 * SYSCALL: RDI,RSI,RDX,R10,R8,R9.
	 *
	 * FAULT: what this side knows about the page, so the monitor can tell
	 * whose fault it is rather than infer it. A page's permissions here
	 * answer to two masters -- the program, whose protections live in the
	 * VMA, and the coherence protocol, which write-protects the PTE and
	 * leaves the VMA alone. PRESENT without WRITABLE on a VMA that permits
	 * writing is this side holding the page for coherence; a write to a VMA
	 * that does not permit writing is the program's own doing.
	 *
	 *   [0] VMCTX_PGS_* flags
	 *   [1] the VMA's protection (PROT_*), 0 if there is no VMA
	 */
	__u64 args[6];
	__u64 fault_addr;	/* FAULT: faulting address                  */
	__u64 fault_err;	/* FAULT: x86 page-fault error code         */
	__u64 rip, rsp, rflags;	/* guest state at the event                 */
};

/*
 * What this side knows about a faulting page. See vmctx_event.args.
 *
 * UNKNOWN is not a failure to answer -- it is the answer, when the fault came
 * from the kernel's own page-fault path and the locks that would have to be
 * taken to look are already held by the caller. A monitor that reads it as
 * "no mapping" would turn every such fault into a segmentation fault.
 */
#define VMCTX_PGS_MAPPED	(1u << 0)  /* a VMA covers the address        */
#define VMCTX_PGS_PRESENT	(1u << 1)  /* a page is mapped in right now   */
#define VMCTX_PGS_WRITABLE	(1u << 2)  /* ...and the PTE permits writing  */
#define VMCTX_PGS_UNKNOWN	(1u << 3)  /* could not be looked at safely   */

/*
 * FAULT events only, args[2]: what kind of mapping the fault was taken in,
 * reported by the mm hook that took it -- the one place the VMA is in hand.
 *
 * The monitor needs this because the right answer to a fault depends on the
 * mapping's class as well as on the ownership record: a page of a private
 * file mapping that was never handed over IS the file, and the local kernel
 * is the right thing to serve it; the same page after a hand-over is on the
 * other machine, and the file here is stale. Without the class the monitor
 * can only choose one policy for both, and both choices were measured wrong
 * (zeros poked over rodata on one arm, ENOENT for a file that exists on the
 * other).
 *
 * VALID says the hook filled this in. Faults reported by a hardware backend
 * carry 0 here -- the guest's mapping class is the monitor's own business on
 * that side -- and a monitor on an older kernel reads 0 and behaves as
 * before.
 */
#define VMCTX_PGC_VALID		(1u << 0)  /* args[2] was filled in at all    */
#define VMCTX_PGC_FILE		(1u << 1)  /* the VMA has a file behind it    */
#define VMCTX_PGC_SHARED	(1u << 2)  /* MAP_SHARED                      */
#define VMCTX_PGC_WRITE		(1u << 3)  /* the VMA permits writing         */

/* reply actions */
#define VMCTX_ACT_SELF		0  /* let the kernel handle it after all      */
#define VMCTX_ACT_DONE		1  /* monitor handled it; SYSCALL uses .retval */
#define VMCTX_ACT_KILL		2  /* terminate the context                   */

/*
 * What a reply may do to the guest's address space, besides answering.
 *
 * A syscall that changes memory is performed on the machine that owns the
 * program, and its effect has to reach the machine the program runs on -- or
 * the two disagree about what is mapped and with what permission. Before this
 * they did: mmap, mprotect and munmap were forwarded to the source, applied to
 * the shadow's mirror, and never applied to the guest at all. The guest's
 * address space was instead conjured a page at a time by the fault path, which
 * maps whatever is touched read-write-execute -- so a PROT_NONE page was
 * readable, a read-only page was writable, and an unmapped address was memory.
 * tests/pf1.c measures exactly that.
 *
 * This is deliberately not a list of syscalls. The monitor says what happened
 * to the address space, in these terms, and the context applies it; which call
 * it came from is the monitor's business. shmat and anything else that maps
 * memory use the same three words.
 */
#define VMCTX_MAP_NONE		0  /* the reply changes no mapping            */
#define VMCTX_MAP_SET		1  /* map [addr,len) from the backing object  */
#define VMCTX_MAP_PROT		2  /* change protection of [addr,len)         */
#define VMCTX_MAP_UNMAP		3  /* remove [addr,len)                       */
/*
 * Map from the *shared* object rather than the context's own. Same three words
 * -- address, length, offset -- and the only difference is which object they
 * are read from, which is the whole of what makes two contexts share a page.
 */
#define VMCTX_MAP_SET_SHARED	4
#define VMCTX_MAP_ZAP		5  /* drop the PAGES of [addr,len), keep the
				    * mappings: the guest-side application of a
				    * source MADV_DONTNEED -- the next touch
				    * faults and refetches the fresh content   */
/*
 * VMCTX_MAP_SET, and the pages of [addr,len) are made PRESENT before the
 * context runs again (MAP_POPULATE): the monitor has just put every one of
 * them into the backing object -- a chunk fetched around one fault -- and a
 * page that is present does not fault, so the fifteen neighbours cost no
 * VMEXIT and no monitor round trip when the program reaches them. The
 * monitor may name ONLY pages it has verified the object holds: populating
 * a hole would have shmem allocate a zero folio in the object (the session
 * 36 door), which is the one thing no fetch path may do.
 */
#define VMCTX_MAP_SET_POPULATE	6

struct vmctx_reply {
	__u32 action;		/* VMCTX_ACT_*                              */
	__u32 _pad;
	__u64 retval;		/* SYSCALL + DONE: value returned to guest   */
	__u32 map_op;		/* VMCTX_MAP_*                              */
	__u32 map_prot;		/* PROT_* for SET and PROT                   */
	__u64 map_addr;
	__u64 map_len;
	/*
	 * Where in the backing object the range comes from.
	 *
	 * The rule everywhere else is that a guest virtual address is its own
	 * offset, which is what lets two contexts started with one object see
	 * the same page at the same address -- a thread, on a machine with no
	 * CLONE_VM. It also makes aliasing impossible: two addresses are two
	 * offsets and therefore two pages, permanently.
	 *
	 * Shared memory is aliasing. Two mappings of one file, in one process
	 * or across a fork, are one page reached by two addresses, and the side
	 * that owns the program is the only one that knows they are the same --
	 * it holds the file. So it says so here, and the ordinary case simply
	 * passes the address back.
	 */
	__u64 map_off;
};

/* Guest registers, as seen and settable by a monitor. */
struct vmctx_uregs {
	__u64 rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp;
	__u64 r8, r9, r10, r11, r12, r13, r14, r15;
	__u64 rip, rflags, orig_rax;
	/*
	 * The thread pointer. pt_regs cannot express it, but a restored or
	 * transplanted context is useless without it: libc reaches everything
	 * thread-local through FS, so a wrong base faults on the first access.
	 * Zero leaves the current value alone.
	 */
	__u64 fs_base, gs_base;
};

/*
 * Which registers the call wrote, as bits of vmctx_syscall.changed.
 *
 * The monitor is handed this rather than left to work it out. Without it the
 * only way to know what a call did to the machine state is to read the whole
 * register set before and after and compare -- two round trips and a diff, per
 * syscall, to discover that almost every call touched exactly one register. The
 * kernel already has both copies at the moment the call returns, so it does the
 * comparison once, where it is free, and says.
 *
 * Most calls report VMCTX_REG_RAX and nothing else. rt_sigreturn reports
 * everything, which is the case that makes the mask worth having: it is the
 * one call whose whole purpose is to replace the machine state.
 */
#define VMCTX_REG_RAX	(1u << 0)
#define VMCTX_REG_RBX	(1u << 1)
#define VMCTX_REG_RCX	(1u << 2)
#define VMCTX_REG_RDX	(1u << 3)
#define VMCTX_REG_RSI	(1u << 4)
#define VMCTX_REG_RDI	(1u << 5)
#define VMCTX_REG_RBP	(1u << 6)
#define VMCTX_REG_RSP	(1u << 7)
#define VMCTX_REG_R8	(1u << 8)
#define VMCTX_REG_R9	(1u << 9)
#define VMCTX_REG_R10	(1u << 10)
#define VMCTX_REG_R11	(1u << 11)
#define VMCTX_REG_R12	(1u << 12)
#define VMCTX_REG_R13	(1u << 13)
#define VMCTX_REG_R14	(1u << 14)
#define VMCTX_REG_R15	(1u << 15)
#define VMCTX_REG_RIP	(1u << 16)
#define VMCTX_REG_RFLAGS (1u << 17)
/*
 * The bases, which are registers like any other and were missing.
 *
 * arch_prctl(ARCH_SET_FS) changes fs_base and nothing else, so the mask read
 * VMCTX_REG_RAX and the caller was never told. The program then computed every
 * thread-local address from a base that was no longer its own, and the next
 * call it made passed a pointer a gigabyte from its own heap.
 */
#define VMCTX_REG_FSBASE (1u << 18)
#define VMCTX_REG_GSBASE (1u << 19)

/*
 * The argument to VMCTX_CTL_SYSCALL.
 *
 * The context makes the call. What it did to its own machine state comes back
 * here, and the *monitor* decides what to do with it: the context's registers
 * are put back as they were before the call, so a monitor that wants the change
 * applied says so with SETREGS. Nothing is imposed on a running context by a
 * call it did not make itself.
 */
struct vmctx_syscall {
	__u64 nr;		/* syscall number                            */
	__u64 args[6];		/* RDI, RSI, RDX, R10, R8, R9                */
	__s64 ret;		/* out: what the syscall returned            */
	__u32 changed;		/* out: VMCTX_REG_* the call wrote           */
	__u32 _pad;
	struct vmctx_uregs regs;/* out: the register set after the call      */
};

struct vmctx_mem {
	__u64 addr;		/* address in the context's address space    */
	__u64 len;		/* bytes (capped at 4096 per call)           */
	__u64 buf;		/* monitor-side buffer                      */
};

#endif /* _UAPI_LINUX_VMCTX_H */
