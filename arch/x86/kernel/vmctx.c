// SPDX-License-Identifier: GPL-2.0
/*
 * vmctx — x86-64 generic syscall dispatch for VM contexts (avm.md project).
 *
 * A VM-context guest traps its syscalls (SYSCALL is disabled in the guest, so
 * it raises #UD). The backend hands us the guest register state and we run the
 * call through the kernel's own dispatch table, x64_sys_call(). There is
 * deliberately no per-syscall knowledge here: everything the kernel implements
 * is reachable, because the syscall handlers read their arguments out of a
 * struct pt_regs and we build one from the guest's registers.
 *
 * Not attempted here (and honestly noted): this path skips
 * syscall_enter_from_user_mode(), so seccomp filters, syscall auditing and
 * ptrace syscall stops do not apply to guest syscalls. Wiring those in is the
 * hardening step before this could be considered a security boundary.
 */
#include <linux/vmctx.h>
#include <linux/kernel.h>
#include <linux/errno.h>
#include <linux/export.h>
#include <linux/nospec.h>
#include <linux/syscalls.h>
#include <asm/ptrace.h>
#include <asm/syscall.h>
#include <asm/unistd.h>
#include <uapi/linux/sched.h>

/*
 * Nothing is refused any more, and the reasons the earlier exceptions existed
 * are worth recording:
 *
 *   fork/clone/clone3 — the child gets its own VM context (vmctx_copy_task),
 *     built from the child's pt_regs, which copy_thread() has already set up
 *     correctly: RAX 0, and RSP either inherited (fork) or the new thread's
 *     stack (CLONE_VM). Threads therefore need no special handling: sharing an
 *     address space is exactly what the guest's CR3 already expresses.
 *   execve — the context is rebuilt at the new program's entry
 *     (vmctx_exec_notify), so the exec'd program runs inside it.
 *   rt_sigreturn — pt_regs *is* the canonical guest register frame, so the
 *     kernel's own restore works unchanged.
 */

/*
 * Dispatch one guest syscall. @regs must be the VM-context task's OWN pt_regs,
 * holding the guest's registers with the syscall number in orig_ax — not a
 * copy. Syscalls read their arguments from this frame, and some (notably
 * rt_sigreturn) *write* it via current_pt_regs(): rt_sigreturn restores the
 * entire pre-signal register set there. Passing the real frame is what lets
 * those work, and lets the backend simply reload the guest from it afterwards.
 */
long vmctx_dispatch_syscall(struct pt_regs *regs)
{
	unsigned int nr;
	long ret;

	nr = (unsigned int)regs->orig_ax;
	if (nr >= NR_syscalls) {
		regs->ax = (unsigned long)-ENOSYS;
		return -ENOSYS;
	}
	nr = array_index_nospec(nr, NR_syscalls);

	ret = x64_sys_call(regs, nr);

	/*
	 * Store the return value. For rt_sigreturn this is the AX the handler
	 * restored from the signal frame, so it is correct there too.
	 */
	regs->ax = (unsigned long)ret;
	return ret;
}
EXPORT_SYMBOL_GPL(vmctx_dispatch_syscall);
