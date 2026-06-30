OPENQASM 2.0;
include "qelib1.inc";
// Minimal Clifford+T circuit: H -> T -> CNOT -> T
qreg q[4];
h q[0];
t q[0];
t q[1];
cx q[0], q[1];
cx q[1], q[2];
cx q[0], q[3];
h q[3];
t q[1];
t q[3];
