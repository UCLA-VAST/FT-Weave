OPENQASM 2.0;
include "qelib1.inc";
// Minimal Clifford+Rz circuit: the STAR counterpart of minimal_clifford_t.qasm.
// STAR injects arbitrary-angle rotations directly, so use rz(theta) rather than t.
qreg q[4];
h q[0];
rz(0.31) q[0];
rz(0.17) q[1];
cx q[0], q[1];
cx q[1], q[2];
cx q[0], q[3];
h q[3];
rz(0.42) q[1];
rz(0.08) q[3];
