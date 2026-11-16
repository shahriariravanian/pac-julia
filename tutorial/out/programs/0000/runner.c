#include <stdlib.h>
#include "pacops.h"
#include "runner.h"

const double scales[] = {
	0.0, 0.0,
	91.188, 0.0,
	0.7071067811865476, 0.0,
	0.0, 0.7071067811865476,
	};

const double factors[] = {
	1.0, 0.0,
	-0.35637333490574913, 0.0,
	0.8462553601411626, 0.0,
	};

void forms(double *p, const double *momenta) {
	p[0] = 0.0 + -1.0 * (1.0 * momenta[0]);
	p[1] = 0.0 + -1.0 * (1.0 * momenta[1]);
	p[2] = 0.0 + -1.0 * (1.0 * momenta[2]);
	p[3] = 0.0 + -1.0 * (1.0 * momenta[3]);
	p[4] = 0.0 + -1.0 * (1.0 * momenta[4]);
	p[5] = 0.0 + -1.0 * (1.0 * momenta[5]);
	p[6] = 0.0 + -1.0 * (1.0 * momenta[6]);
	p[7] = 0.0 + -1.0 * (1.0 * momenta[7]);
	p[8] = 0.0 + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[8]);
	p[9] = 0.0 + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[9]);
	p[10] = 0.0 + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[10]);
	p[11] = 0.0 + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[11]);
	p[12] = 0.0 + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[8]) + 1.0 * (1.0 * momenta[12]);
	p[13] = 0.0 + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[9]) + 1.0 * (1.0 * momenta[13]);
	p[14] = 0.0 + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[10]) + 1.0 * (1.0 * momenta[14]);
	p[15] = 0.0 + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[11]) + 1.0 * (1.0 * momenta[15]);
	p[16] = 0.0 + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[8]) + 1.0 * (1.0 * momenta[12]) + 1.0 * (1.0 * momenta[16]);
	p[17] = 0.0 + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[9]) + 1.0 * (1.0 * momenta[13]) + 1.0 * (1.0 * momenta[17]);
	p[18] = 0.0 + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[10]) + 1.0 * (1.0 * momenta[14]) + 1.0 * (1.0 * momenta[18]);
	p[19] = 0.0 + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[11]) + 1.0 * (1.0 * momenta[15]) + 1.0 * (1.0 * momenta[19]);
	p[20] = 0.0 + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[12]);
	p[21] = 0.0 + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[13]);
	p[22] = 0.0 + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[14]);
	p[23] = 0.0 + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[15]);
	p[24] = 0.0 + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[12]) + 1.0 * (1.0 * momenta[16]);
	p[25] = 0.0 + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[13]) + 1.0 * (1.0 * momenta[17]);
	p[26] = 0.0 + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[14]) + 1.0 * (1.0 * momenta[18]);
	p[27] = 0.0 + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[15]) + 1.0 * (1.0 * momenta[19]);
	p[28] = 0.0 + 1.0 * (1.0 * momenta[8]);
	p[29] = 0.0 + 1.0 * (1.0 * momenta[9]);
	p[30] = 0.0 + 1.0 * (1.0 * momenta[10]);
	p[31] = 0.0 + 1.0 * (1.0 * momenta[11]);
	p[32] = 0.0 + 1.0 * (1.0 * momenta[12]);
	p[33] = 0.0 + 1.0 * (1.0 * momenta[13]);
	p[34] = 0.0 + 1.0 * (1.0 * momenta[14]);
	p[35] = 0.0 + 1.0 * (1.0 * momenta[15]);
	p[36] = 0.0 + 1.0 * (1.0 * momenta[12]) + 1.0 * (1.0 * momenta[16]);
	p[37] = 0.0 + 1.0 * (1.0 * momenta[13]) + 1.0 * (1.0 * momenta[17]);
	p[38] = 0.0 + 1.0 * (1.0 * momenta[14]) + 1.0 * (1.0 * momenta[18]);
	p[39] = 0.0 + 1.0 * (1.0 * momenta[15]) + 1.0 * (1.0 * momenta[19]);
	p[40] = 0.0 + 1.0 * (1.0 * momenta[16]);
	p[41] = 0.0 + 1.0 * (1.0 * momenta[17]);
	p[42] = 0.0 + 1.0 * (1.0 * momenta[18]);
	p[43] = 0.0 + 1.0 * (1.0 * momenta[19]);
	}

static const double cf1[] = {1.0, 0.0, 1.0, 0.0};

void body(double *out, double *p) {
	// 1: CLEAR_AMPLITUDE a0
	p[44] = 0.0; p[45] = 0.0;
	// 2: CLEAR_AMPLITUDE a1
	p[46] = 0.0; p[47] = 0.0;
	// 3: CLEAR_AMPLITUDE a2
	p[48] = 0.0; p[49] = 0.0;
	// 4: CLEAR_AMPLITUDE a3
	p[50] = 0.0; p[51] = 0.0;
	// 5: CLEAR_AMPLITUDE a4
	p[52] = 0.0; p[53] = 0.0;
	// 6: CLEAR_AMPLITUDE a5
	p[54] = 0.0; p[55] = 0.0;
	// 7: CLEAR_AMPLITUDE a6
	p[56] = 0.0; p[57] = 0.0;
	// 8: CLEAR_AMPLITUDE a7
	p[58] = 0.0; p[59] = 0.0;
	// 9: CLEAR_AMPLITUDE a8
	p[60] = 0.0; p[61] = 0.0;
	// 10: CLEAR_AMPLITUDE a9
	p[62] = 0.0; p[63] = 0.0;
	// 11: CLEAR_AMPLITUDE a10
	p[64] = 0.0; p[65] = 0.0;
	// 12: CLEAR_AMPLITUDE a11
	p[66] = 0.0; p[67] = 0.0;
	// 13: CLEAR_AMPLITUDE a12
	p[68] = 0.0; p[69] = 0.0;
	// 14: CLEAR_AMPLITUDE a13
	p[70] = 0.0; p[71] = 0.0;
	// 15: CLEAR_AMPLITUDE a14
	p[72] = 0.0; p[73] = 0.0;
	// 16: CLEAR_AMPLITUDE a15
	p[74] = 0.0; p[75] = 0.0;
	// 17: CLEAR_AMPLITUDE a16
	p[76] = 0.0; p[77] = 0.0;
	// 18: CLEAR_AMPLITUDE a17
	p[78] = 0.0; p[79] = 0.0;
	// 19: CLEAR_AMPLITUDE a18
	p[80] = 0.0; p[81] = 0.0;
	// 20: CLEAR_AMPLITUDE a19
	p[82] = 0.0; p[83] = 0.0;
	// 21: CLEAR_AMPLITUDE a20
	p[84] = 0.0; p[85] = 0.0;
	// 22: CLEAR_AMPLITUDE a21
	p[86] = 0.0; p[87] = 0.0;
	// 23: CLEAR_AMPLITUDE a22
	p[88] = 0.0; p[89] = 0.0;
	// 24: CLEAR_AMPLITUDE a23
	p[90] = 0.0; p[91] = 0.0;
	// 25: SOURCE c0@q0 = WEYL antiparticle, helicity -1, chirality 1, mass s0; x0
	source(&p[92], FAMILY_WEYL, ANTIPARTICLE, &p[0], -1, 1, 0.0, &factors[0]);
	// 26: SOURCE c1@q7 = VECTOR, helicity -1, chirality 0, mass s1; x0
	source(&p[108], FAMILY_VECTOR, PARTICLE, &p[28], -1, 0, 91.188, &factors[0]);
	// 27: SOURCE c2@q7 = VECTOR, helicity 0, chirality 0, mass s1; x0
	source(&p[116], FAMILY_VECTOR, PARTICLE, &p[28], 0, 0, 91.188, &factors[0]);
	// 28: SOURCE c3@q1 = WEYL, helicity -1, chirality 1, mass s0; x0
	source(&p[100], FAMILY_WEYL, PARTICLE, &p[4], -1, 1, 0.0, &factors[0]);
	// 29: SOURCE c4@q7 = VECTOR, helicity 1, chirality 0, mass s1; x0
	source(&p[124], FAMILY_VECTOR, PARTICLE, &p[28], 1, 0, 91.188, &factors[0]);
	// 30: SOURCE c5@q0 = WEYL antiparticle, helicity 1, chirality -1, mass s0; x0
	source(&p[96], FAMILY_WEYL, ANTIPARTICLE, &p[0], 1, -1, 0.0, &factors[0]);
	// 31: SOURCE c6@q8 = VECTOR, helicity -1, chirality 0, mass s0; x0
	source(&p[132], FAMILY_VECTOR, PARTICLE, &p[32], -1, 0, 0.0, &factors[0]);
	// 32: SOURCE c7@q10 = VECTOR, helicity -1, chirality 0, mass s0; x0
	source(&p[148], FAMILY_VECTOR, PARTICLE, &p[40], -1, 0, 0.0, &factors[0]);
	// 33: SOURCE c8@q10 = VECTOR, helicity 1, chirality 0, mass s0; x0
	source(&p[156], FAMILY_VECTOR, PARTICLE, &p[40], 1, 0, 0.0, &factors[0]);
	// 34: SOURCE c9@q8 = VECTOR, helicity 1, chirality 0, mass s0; x0
	source(&p[140], FAMILY_VECTOR, PARTICLE, &p[32], 1, 0, 0.0, &factors[0]);
	// 35: SOURCE c10@q1 = WEYL, helicity 1, chirality -1, mass s0; x0
	source(&p[104], FAMILY_WEYL, PARTICLE, &p[4], 1, -1, 0.0, &factors[0]);
	// 36: CLEAR c11, 2 components
	clear(&p[164], 2);
	// 37: CLEAR c12, 2 components
	clear(&p[168], 2);
	// 38: CLEAR c13, 2 components
	clear(&p[172], 2);
	// 39: CLEAR c14, 2 components
	clear(&p[176], 2);
	// 40: CLEAR c15, 2 components
	clear(&p[180], 2);
	// 41: CLEAR c16, 2 components
	clear(&p[184], 2);
	// 42: CLEAR c17, 2 components
	clear(&p[188], 2);
	// 43: CLEAR c18, 2 components
	clear(&p[192], 2);
	// 44: CLEAR c19, 2 components
	clear(&p[196], 2);
	// 45: CLEAR c20, 2 components
	clear(&p[200], 2);
	// 46: CLEAR c21, 4 components
	clear(&p[204], 4);
	// 47: CLEAR c22, 4 components
	clear(&p[212], 4);
	// 48: CLEAR c23, 4 components
	clear(&p[220], 4);
	// 49: CLEAR c24, 4 components
	clear(&p[228], 4);
	// 50: V_WEYL_VECTOR_A c27@q2 = c3@q1, c1@q7; s2 x1
	vertex(&p[164], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[108], NULL, NULL, &scales[4], &factors[2]);
	// 51: V_WEYL_VECTOR_A c28@q2 = c3@q1, c2@q7; s2 x1
	vertex(&p[168], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[116], NULL, NULL, &scales[4], &factors[2]);
	// 52: V_WEYL_VECTOR_A c29@q2 = c3@q1, c4@q7; s2 x1
	vertex(&p[172], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[124], NULL, NULL, &scales[4], &factors[2]);
	// 53: V_WEYL_VECTOR_A c33@q5 = c3@q1, c6@q8; s2 x0
	vertex(&p[176], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[132], NULL, NULL, &scales[4], &factors[0]);
	// 54: V_WEYL_VECTOR_A c34@q5 = c3@q1, c9@q8; s2 x0
	vertex(&p[180], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[140], NULL, NULL, &scales[4], &factors[0]);
	// 55: V_WEYL_VECTOR_B c30@q2 = c10@q1, c1@q7; s2 x2
	vertex(&p[184], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[108], NULL, NULL, &scales[4], &factors[4]);
	// 56: V_WEYL_VECTOR_B c31@q2 = c10@q1, c2@q7; s2 x2
	vertex(&p[188], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[116], NULL, NULL, &scales[4], &factors[4]);
	// 57: V_WEYL_VECTOR_B c32@q2 = c10@q1, c4@q7; s2 x2
	vertex(&p[192], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[124], NULL, NULL, &scales[4], &factors[4]);
	// 58: V_WEYL_VECTOR_B c25@q5 = c10@q1, c6@q8; s2 x0
	vertex(&p[196], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[132], NULL, NULL, &scales[4], &factors[0]);
	// 59: V_WEYL_VECTOR_B c26@q5 = c10@q1, c9@q8; s2 x0
	vertex(&p[200], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[140], NULL, NULL, &scales[4], &factors[0]);
	// 60: V_THREE_VECTOR c35@q9 = c6@q8, c7@q10; s3 x0
	vertex(&p[204], VERTEX_THREE_VECTOR, MODE_SET, &p[132], &p[148], &p[32], &p[40], &scales[6], &factors[0]);
	// 61: V_THREE_VECTOR c36@q9 = c6@q8, c8@q10; s3 x0
	vertex(&p[212], VERTEX_THREE_VECTOR, MODE_SET, &p[132], &p[156], &p[32], &p[40], &scales[6], &factors[0]);
	// 62: V_THREE_VECTOR c37@q9 = c9@q8, c7@q10; s3 x0
	vertex(&p[220], VERTEX_THREE_VECTOR, MODE_SET, &p[140], &p[148], &p[32], &p[40], &scales[6], &factors[0]);
	// 63: V_THREE_VECTOR c38@q9 = c9@q8, c8@q10; s3 x0
	vertex(&p[228], VERTEX_THREE_VECTOR, MODE_SET, &p[140], &p[156], &p[32], &p[40], &scales[6], &factors[0]);
	// 64: P_WEYL_A c30@q2; x0
	propagator(&p[184], PROPAGATOR_WEYL_A, &p[184], &p[8], &factors[0]);
	// 65: P_WEYL_A c31@q2; x0
	propagator(&p[188], PROPAGATOR_WEYL_A, &p[188], &p[8], &factors[0]);
	// 66: P_WEYL_A c25@q5; x0
	propagator(&p[196], PROPAGATOR_WEYL_A, &p[196], &p[20], &factors[0]);
	// 67: P_WEYL_A c26@q5; x0
	propagator(&p[200], PROPAGATOR_WEYL_A, &p[200], &p[20], &factors[0]);
	// 68: P_WEYL_A c32@q2; x0
	propagator(&p[192], PROPAGATOR_WEYL_A, &p[192], &p[8], &factors[0]);
	// 69: P_WEYL_B c27@q2; x0
	propagator(&p[164], PROPAGATOR_WEYL_B, &p[164], &p[8], &factors[0]);
	// 70: P_WEYL_B c28@q2; x0
	propagator(&p[168], PROPAGATOR_WEYL_B, &p[168], &p[8], &factors[0]);
	// 71: P_WEYL_B c33@q5; x0
	propagator(&p[176], PROPAGATOR_WEYL_B, &p[176], &p[20], &factors[0]);
	// 72: P_WEYL_B c34@q5; x0
	propagator(&p[180], PROPAGATOR_WEYL_B, &p[180], &p[20], &factors[0]);
	// 73: P_WEYL_B c29@q2; x0
	propagator(&p[172], PROPAGATOR_WEYL_B, &p[172], &p[8], &factors[0]);
	// 74: P_VECTOR c35@q9; x0
	propagator(&p[204], PROPAGATOR_VECTOR, &p[204], &p[36], &factors[0]);
	// 75: P_VECTOR c36@q9; x0
	propagator(&p[212], PROPAGATOR_VECTOR, &p[212], &p[36], &factors[0]);
	// 76: P_VECTOR c37@q9; x0
	propagator(&p[220], PROPAGATOR_VECTOR, &p[220], &p[36], &factors[0]);
	// 77: P_VECTOR c38@q9; x0
	propagator(&p[228], PROPAGATOR_VECTOR, &p[228], &p[36], &factors[0]);
	// 78: CLEAR c39, 2 components
	clear(&p[236], 2);
	// 79: CLEAR c40, 2 components
	clear(&p[240], 2);
	// 80: CLEAR c41, 2 components
	clear(&p[244], 2);
	// 81: CLEAR c42, 2 components
	clear(&p[248], 2);
	// 82: CLEAR c43, 2 components
	clear(&p[252], 2);
	// 83: CLEAR c44, 2 components
	clear(&p[256], 2);
	// 84: CLEAR c45, 2 components
	clear(&p[260], 2);
	// 85: CLEAR c46, 2 components
	clear(&p[264], 2);
	// 86: CLEAR c47, 2 components
	clear(&p[268], 2);
	// 87: CLEAR c48, 2 components
	clear(&p[272], 2);
	// 88: CLEAR c49, 2 components
	clear(&p[276], 2);
	// 89: CLEAR c50, 2 components
	clear(&p[280], 2);
	// 90: CLEAR c51, 2 components
	clear(&p[284], 2);
	// 91: CLEAR c52, 2 components
	clear(&p[288], 2);
	// 92: CLEAR c53, 2 components
	clear(&p[292], 2);
	// 93: CLEAR c54, 2 components
	clear(&p[296], 2);
	// 94: CLEAR c55, 2 components
	clear(&p[300], 2);
	// 95: CLEAR c56, 2 components
	clear(&p[304], 2);
	// 96: CLEAR c57, 2 components
	clear(&p[308], 2);
	// 97: CLEAR c58, 2 components
	clear(&p[312], 2);
	// 98: V_WEYL_VECTOR_A c75@q6 = c3@q1, c35@q9; s2 x0 [run starts]
	vertex(&p[236], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[204], NULL, NULL, &scales[4], &factors[0]);
	// 99: V_WEYL_VECTOR_A c75@q6 += c33@q5, c7@q10; s2 x0 [run ends]
	vertex(&p[236], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[176], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 100: V_WEYL_VECTOR_A c76@q6 = c3@q1, c36@q9; s2 x0 [run starts]
	vertex(&p[240], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[212], NULL, NULL, &scales[4], &factors[0]);
	// 101: V_WEYL_VECTOR_A c76@q6 += c33@q5, c8@q10; s2 x0 [run ends]
	vertex(&p[240], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[176], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 102: V_WEYL_VECTOR_A c77@q6 = c3@q1, c37@q9; s2 x0 [run starts]
	vertex(&p[244], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[220], NULL, NULL, &scales[4], &factors[0]);
	// 103: V_WEYL_VECTOR_A c77@q6 += c34@q5, c7@q10; s2 x0 [run ends]
	vertex(&p[244], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[180], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 104: V_WEYL_VECTOR_A c78@q6 = c3@q1, c38@q9; s2 x0 [run starts]
	vertex(&p[248], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[100], &p[228], NULL, NULL, &scales[4], &factors[0]);
	// 105: V_WEYL_VECTOR_A c78@q6 += c34@q5, c8@q10; s2 x0 [run ends]
	vertex(&p[248], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[180], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 106: V_WEYL_VECTOR_B c63@q6 = c10@q1, c35@q9; s2 x0 [run starts]
	vertex(&p[252], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[204], NULL, NULL, &scales[4], &factors[0]);
	// 107: V_WEYL_VECTOR_B c63@q6 += c25@q5, c7@q10; s2 x0 [run ends]
	vertex(&p[252], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[196], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 108: V_WEYL_VECTOR_B c64@q6 = c10@q1, c36@q9; s2 x0 [run starts]
	vertex(&p[256], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[212], NULL, NULL, &scales[4], &factors[0]);
	// 109: V_WEYL_VECTOR_B c64@q6 += c25@q5, c8@q10; s2 x0 [run ends]
	vertex(&p[256], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[196], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 110: V_WEYL_VECTOR_B c66@q6 = c10@q1, c37@q9; s2 x0 [run starts]
	vertex(&p[260], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[220], NULL, NULL, &scales[4], &factors[0]);
	// 111: V_WEYL_VECTOR_B c66@q6 += c26@q5, c7@q10; s2 x0 [run ends]
	vertex(&p[260], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[200], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 112: V_WEYL_VECTOR_B c67@q6 = c10@q1, c38@q9; s2 x0 [run starts]
	vertex(&p[264], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[104], &p[228], NULL, NULL, &scales[4], &factors[0]);
	// 113: V_WEYL_VECTOR_B c67@q6 += c26@q5, c8@q10; s2 x0 [run ends]
	vertex(&p[264], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[200], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 114: V_WEYL_VECTOR_A c69@q3 = c33@q5, c1@q7; s2 x1 [run starts]
	vertex(&p[268], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[176], &p[108], NULL, NULL, &scales[4], &factors[2]);
	// 115: V_WEYL_VECTOR_A c69@q3 += c27@q2, c6@q8; s2 x0 [run ends]
	vertex(&p[268], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[164], &p[132], NULL, NULL, &scales[4], &factors[0]);
	// 116: V_WEYL_VECTOR_A c70@q3 = c34@q5, c1@q7; s2 x1 [run starts]
	vertex(&p[272], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[180], &p[108], NULL, NULL, &scales[4], &factors[2]);
	// 117: V_WEYL_VECTOR_A c70@q3 += c27@q2, c9@q8; s2 x0 [run ends]
	vertex(&p[272], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[164], &p[140], NULL, NULL, &scales[4], &factors[0]);
	// 118: V_WEYL_VECTOR_B c59@q3 = c30@q2, c6@q8; s2 x0 [run starts]
	vertex(&p[276], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[184], &p[132], NULL, NULL, &scales[4], &factors[0]);
	// 119: V_WEYL_VECTOR_B c59@q3 += c25@q5, c1@q7; s2 x2 [run ends]
	vertex(&p[276], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[196], &p[108], NULL, NULL, &scales[4], &factors[4]);
	// 120: V_WEYL_VECTOR_B c60@q3 = c30@q2, c9@q8; s2 x0 [run starts]
	vertex(&p[280], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[184], &p[140], NULL, NULL, &scales[4], &factors[0]);
	// 121: V_WEYL_VECTOR_B c60@q3 += c26@q5, c1@q7; s2 x2 [run ends]
	vertex(&p[280], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[200], &p[108], NULL, NULL, &scales[4], &factors[4]);
	// 122: V_WEYL_VECTOR_A c71@q3 = c33@q5, c2@q7; s2 x1 [run starts]
	vertex(&p[284], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[176], &p[116], NULL, NULL, &scales[4], &factors[2]);
	// 123: V_WEYL_VECTOR_A c71@q3 += c28@q2, c6@q8; s2 x0 [run ends]
	vertex(&p[284], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[168], &p[132], NULL, NULL, &scales[4], &factors[0]);
	// 124: V_WEYL_VECTOR_A c72@q3 = c34@q5, c2@q7; s2 x1 [run starts]
	vertex(&p[288], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[180], &p[116], NULL, NULL, &scales[4], &factors[2]);
	// 125: V_WEYL_VECTOR_A c72@q3 += c28@q2, c9@q8; s2 x0 [run ends]
	vertex(&p[288], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[168], &p[140], NULL, NULL, &scales[4], &factors[0]);
	// 126: V_WEYL_VECTOR_B c61@q3 = c31@q2, c6@q8; s2 x0 [run starts]
	vertex(&p[292], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[188], &p[132], NULL, NULL, &scales[4], &factors[0]);
	// 127: V_WEYL_VECTOR_B c61@q3 += c25@q5, c2@q7; s2 x2 [run ends]
	vertex(&p[292], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[196], &p[116], NULL, NULL, &scales[4], &factors[4]);
	// 128: V_WEYL_VECTOR_B c62@q3 = c31@q2, c9@q8; s2 x0 [run starts]
	vertex(&p[296], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[188], &p[140], NULL, NULL, &scales[4], &factors[0]);
	// 129: V_WEYL_VECTOR_B c62@q3 += c26@q5, c2@q7; s2 x2 [run ends]
	vertex(&p[296], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[200], &p[116], NULL, NULL, &scales[4], &factors[4]);
	// 130: V_WEYL_VECTOR_A c73@q3 = c33@q5, c4@q7; s2 x1 [run starts]
	vertex(&p[300], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[176], &p[124], NULL, NULL, &scales[4], &factors[2]);
	// 131: V_WEYL_VECTOR_A c73@q3 += c29@q2, c6@q8; s2 x0 [run ends]
	vertex(&p[300], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[172], &p[132], NULL, NULL, &scales[4], &factors[0]);
	// 132: V_WEYL_VECTOR_A c74@q3 = c34@q5, c4@q7; s2 x1 [run starts]
	vertex(&p[304], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[180], &p[124], NULL, NULL, &scales[4], &factors[2]);
	// 133: V_WEYL_VECTOR_A c74@q3 += c29@q2, c9@q8; s2 x0 [run ends]
	vertex(&p[304], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[172], &p[140], NULL, NULL, &scales[4], &factors[0]);
	// 134: V_WEYL_VECTOR_B c65@q3 = c32@q2, c6@q8; s2 x0 [run starts]
	vertex(&p[308], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[192], &p[132], NULL, NULL, &scales[4], &factors[0]);
	// 135: V_WEYL_VECTOR_B c65@q3 += c25@q5, c4@q7; s2 x2 [run ends]
	vertex(&p[308], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[196], &p[124], NULL, NULL, &scales[4], &factors[4]);
	// 136: V_WEYL_VECTOR_B c68@q3 = c32@q2, c9@q8; s2 x0 [run starts]
	vertex(&p[312], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[192], &p[140], NULL, NULL, &scales[4], &factors[0]);
	// 137: V_WEYL_VECTOR_B c68@q3 += c26@q5, c4@q7; s2 x2 [run ends]
	vertex(&p[312], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[200], &p[124], NULL, NULL, &scales[4], &factors[4]);
	// 138: P_WEYL_A c59@q3; x0
	propagator(&p[276], PROPAGATOR_WEYL_A, &p[276], &p[12], &factors[0]);
	// 139: P_WEYL_A c60@q3; x0
	propagator(&p[280], PROPAGATOR_WEYL_A, &p[280], &p[12], &factors[0]);
	// 140: P_WEYL_A c61@q3; x0
	propagator(&p[292], PROPAGATOR_WEYL_A, &p[292], &p[12], &factors[0]);
	// 141: P_WEYL_A c62@q3; x0
	propagator(&p[296], PROPAGATOR_WEYL_A, &p[296], &p[12], &factors[0]);
	// 142: P_WEYL_A c63@q6; x0
	propagator(&p[252], PROPAGATOR_WEYL_A, &p[252], &p[24], &factors[0]);
	// 143: P_WEYL_A c64@q6; x0
	propagator(&p[256], PROPAGATOR_WEYL_A, &p[256], &p[24], &factors[0]);
	// 144: P_WEYL_A c65@q3; x0
	propagator(&p[308], PROPAGATOR_WEYL_A, &p[308], &p[12], &factors[0]);
	// 145: P_WEYL_A c66@q6; x0
	propagator(&p[260], PROPAGATOR_WEYL_A, &p[260], &p[24], &factors[0]);
	// 146: P_WEYL_A c67@q6; x0
	propagator(&p[264], PROPAGATOR_WEYL_A, &p[264], &p[24], &factors[0]);
	// 147: P_WEYL_A c68@q3; x0
	propagator(&p[312], PROPAGATOR_WEYL_A, &p[312], &p[12], &factors[0]);
	// 148: P_WEYL_B c69@q3; x0
	propagator(&p[268], PROPAGATOR_WEYL_B, &p[268], &p[12], &factors[0]);
	// 149: P_WEYL_B c70@q3; x0
	propagator(&p[272], PROPAGATOR_WEYL_B, &p[272], &p[12], &factors[0]);
	// 150: P_WEYL_B c71@q3; x0
	propagator(&p[284], PROPAGATOR_WEYL_B, &p[284], &p[12], &factors[0]);
	// 151: P_WEYL_B c72@q3; x0
	propagator(&p[288], PROPAGATOR_WEYL_B, &p[288], &p[12], &factors[0]);
	// 152: P_WEYL_B c75@q6; x0
	propagator(&p[236], PROPAGATOR_WEYL_B, &p[236], &p[24], &factors[0]);
	// 153: P_WEYL_B c76@q6; x0
	propagator(&p[240], PROPAGATOR_WEYL_B, &p[240], &p[24], &factors[0]);
	// 154: P_WEYL_B c73@q3; x0
	propagator(&p[300], PROPAGATOR_WEYL_B, &p[300], &p[12], &factors[0]);
	// 155: P_WEYL_B c77@q6; x0
	propagator(&p[244], PROPAGATOR_WEYL_B, &p[244], &p[24], &factors[0]);
	// 156: P_WEYL_B c78@q6; x0
	propagator(&p[248], PROPAGATOR_WEYL_B, &p[248], &p[24], &factors[0]);
	// 157: P_WEYL_B c74@q3; x0
	propagator(&p[304], PROPAGATOR_WEYL_B, &p[304], &p[12], &factors[0]);
	// 158: CLEAR c79, 2 components
	clear(&p[100], 2);
	// 159: CLEAR c80, 2 components
	clear(&p[104], 2);
	// 160: CLEAR c14, 2 components
	clear(&p[176], 2);
	// 161: CLEAR c15, 2 components
	clear(&p[180], 2);
	// 162: CLEAR c19, 2 components
	clear(&p[196], 2);
	// 163: CLEAR c20, 2 components
	clear(&p[200], 2);
	// 164: CLEAR c81, 2 components
	clear(&p[132], 2);
	// 165: CLEAR c82, 2 components
	clear(&p[136], 2);
	// 166: CLEAR c83, 2 components
	clear(&p[140], 2);
	// 167: CLEAR c84, 2 components
	clear(&p[144], 2);
	// 168: CLEAR c85, 2 components
	clear(&p[316], 2);
	// 169: CLEAR c86, 2 components
	clear(&p[320], 2);
	// 170: CLEAR c87, 2 components
	clear(&p[324], 2);
	// 171: CLEAR c88, 2 components
	clear(&p[328], 2);
	// 172: CLEAR c89, 2 components
	clear(&p[332], 2);
	// 173: CLEAR c90, 2 components
	clear(&p[336], 2);
	// 174: CLEAR c91, 2 components
	clear(&p[340], 2);
	// 175: CLEAR c92, 2 components
	clear(&p[344], 2);
	// 176: CLEAR c93, 2 components
	clear(&p[348], 2);
	// 177: CLEAR c94, 2 components
	clear(&p[352], 2);
	// 178: CLEAR c95, 2 components
	clear(&p[356], 2);
	// 179: CLEAR c96, 2 components
	clear(&p[360], 2);
	// 180: CLEAR c97, 2 components
	clear(&p[364], 2);
	// 181: CLEAR c98, 2 components
	clear(&p[368], 2);
	// 182: V_WEYL_VECTOR_A c3@q1 = c75@q6, c1@q7; s2 x1 [run starts]
	vertex(&p[100], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[236], &p[108], NULL, NULL, &scales[4], &factors[2]);
	// 183: V_WEYL_VECTOR_A c3@q1 += c27@q2, c35@q9; s2 x0 [in run]
	vertex(&p[100], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[164], &p[204], NULL, NULL, &scales[4], &factors[0]);
	// 184: V_WEYL_VECTOR_A c3@q1 += c69@q3, c7@q10; s2 x0 [run ends]
	vertex(&p[100], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[268], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 185: V_WEYL_VECTOR_A c10@q1 = c76@q6, c1@q7; s2 x1 [run starts]
	vertex(&p[104], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[240], &p[108], NULL, NULL, &scales[4], &factors[2]);
	// 186: V_WEYL_VECTOR_A c10@q1 += c27@q2, c36@q9; s2 x0 [in run]
	vertex(&p[104], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[164], &p[212], NULL, NULL, &scales[4], &factors[0]);
	// 187: V_WEYL_VECTOR_A c10@q1 += c69@q3, c8@q10; s2 x0 [run ends]
	vertex(&p[104], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[268], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 188: V_WEYL_VECTOR_B c99 = c30@q2, c37@q9; s2 x0 [run starts]
	vertex(&p[132], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[184], &p[220], NULL, NULL, &scales[4], &factors[0]);
	// 189: V_WEYL_VECTOR_B c99 += c60@q3, c7@q10; s2 x0 [in run]
	vertex(&p[132], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[280], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 190: V_WEYL_VECTOR_B c99 += c66@q6, c1@q7; s2 x2 [run ends]
	vertex(&p[132], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[260], &p[108], NULL, NULL, &scales[4], &factors[4]);
	// 191: V_WEYL_VECTOR_B c100 = c30@q2, c38@q9; s2 x0 [run starts]
	vertex(&p[136], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[184], &p[228], NULL, NULL, &scales[4], &factors[0]);
	// 192: V_WEYL_VECTOR_B c100 += c60@q3, c8@q10; s2 x0 [in run]
	vertex(&p[136], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[280], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 193: V_WEYL_VECTOR_B c100 += c67@q6, c1@q7; s2 x2 [run ends]
	vertex(&p[136], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[264], &p[108], NULL, NULL, &scales[4], &factors[4]);
	// 194: V_WEYL_VECTOR_A c109 = c75@q6, c2@q7; s2 x1 [run starts]
	vertex(&p[140], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[236], &p[116], NULL, NULL, &scales[4], &factors[2]);
	// 195: V_WEYL_VECTOR_A c109 += c28@q2, c35@q9; s2 x0 [in run]
	vertex(&p[140], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[168], &p[204], NULL, NULL, &scales[4], &factors[0]);
	// 196: V_WEYL_VECTOR_A c109 += c71@q3, c7@q10; s2 x0 [run ends]
	vertex(&p[140], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[284], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 197: V_WEYL_VECTOR_A c110 = c76@q6, c2@q7; s2 x1 [run starts]
	vertex(&p[144], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[240], &p[116], NULL, NULL, &scales[4], &factors[2]);
	// 198: V_WEYL_VECTOR_A c110 += c28@q2, c36@q9; s2 x0 [in run]
	vertex(&p[144], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[168], &p[212], NULL, NULL, &scales[4], &factors[0]);
	// 199: V_WEYL_VECTOR_A c110 += c71@q3, c8@q10; s2 x0 [run ends]
	vertex(&p[144], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[284], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 200: V_WEYL_VECTOR_A c33@q5 = c77@q6, c1@q7; s2 x1 [run starts]
	vertex(&p[176], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[244], &p[108], NULL, NULL, &scales[4], &factors[2]);
	// 201: V_WEYL_VECTOR_A c33@q5 += c27@q2, c37@q9; s2 x0 [in run]
	vertex(&p[176], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[164], &p[220], NULL, NULL, &scales[4], &factors[0]);
	// 202: V_WEYL_VECTOR_A c33@q5 += c70@q3, c7@q10; s2 x0 [run ends]
	vertex(&p[176], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[272], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 203: V_WEYL_VECTOR_A c34@q5 = c78@q6, c1@q7; s2 x1 [run starts]
	vertex(&p[180], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[248], &p[108], NULL, NULL, &scales[4], &factors[2]);
	// 204: V_WEYL_VECTOR_A c34@q5 += c27@q2, c38@q9; s2 x0 [in run]
	vertex(&p[180], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[164], &p[228], NULL, NULL, &scales[4], &factors[0]);
	// 205: V_WEYL_VECTOR_A c34@q5 += c70@q3, c8@q10; s2 x0 [run ends]
	vertex(&p[180], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[272], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 206: V_WEYL_VECTOR_B c25@q5 = c30@q2, c35@q9; s2 x0 [run starts]
	vertex(&p[196], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[184], &p[204], NULL, NULL, &scales[4], &factors[0]);
	// 207: V_WEYL_VECTOR_B c25@q5 += c59@q3, c7@q10; s2 x0 [in run]
	vertex(&p[196], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[276], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 208: V_WEYL_VECTOR_B c25@q5 += c63@q6, c1@q7; s2 x2 [run ends]
	vertex(&p[196], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[252], &p[108], NULL, NULL, &scales[4], &factors[4]);
	// 209: V_WEYL_VECTOR_B c26@q5 = c30@q2, c36@q9; s2 x0 [run starts]
	vertex(&p[200], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[184], &p[212], NULL, NULL, &scales[4], &factors[0]);
	// 210: V_WEYL_VECTOR_B c26@q5 += c59@q3, c8@q10; s2 x0 [in run]
	vertex(&p[200], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[276], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 211: V_WEYL_VECTOR_B c26@q5 += c64@q6, c1@q7; s2 x2 [run ends]
	vertex(&p[200], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[256], &p[108], NULL, NULL, &scales[4], &factors[4]);
	// 212: V_WEYL_VECTOR_A c111 = c77@q6, c2@q7; s2 x1 [run starts]
	vertex(&p[316], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[244], &p[116], NULL, NULL, &scales[4], &factors[2]);
	// 213: V_WEYL_VECTOR_A c111 += c28@q2, c37@q9; s2 x0 [in run]
	vertex(&p[316], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[168], &p[220], NULL, NULL, &scales[4], &factors[0]);
	// 214: V_WEYL_VECTOR_A c111 += c72@q3, c7@q10; s2 x0 [run ends]
	vertex(&p[316], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[288], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 215: V_WEYL_VECTOR_A c112 = c78@q6, c2@q7; s2 x1 [run starts]
	vertex(&p[320], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[248], &p[116], NULL, NULL, &scales[4], &factors[2]);
	// 216: V_WEYL_VECTOR_A c112 += c28@q2, c38@q9; s2 x0 [in run]
	vertex(&p[320], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[168], &p[228], NULL, NULL, &scales[4], &factors[0]);
	// 217: V_WEYL_VECTOR_A c112 += c72@q3, c8@q10; s2 x0 [run ends]
	vertex(&p[320], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[288], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 218: V_WEYL_VECTOR_B c101 = c31@q2, c35@q9; s2 x0 [run starts]
	vertex(&p[324], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[188], &p[204], NULL, NULL, &scales[4], &factors[0]);
	// 219: V_WEYL_VECTOR_B c101 += c61@q3, c7@q10; s2 x0 [in run]
	vertex(&p[324], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[292], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 220: V_WEYL_VECTOR_B c101 += c63@q6, c2@q7; s2 x2 [run ends]
	vertex(&p[324], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[252], &p[116], NULL, NULL, &scales[4], &factors[4]);
	// 221: V_WEYL_VECTOR_B c102 = c31@q2, c36@q9; s2 x0 [run starts]
	vertex(&p[328], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[188], &p[212], NULL, NULL, &scales[4], &factors[0]);
	// 222: V_WEYL_VECTOR_B c102 += c61@q3, c8@q10; s2 x0 [in run]
	vertex(&p[328], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[292], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 223: V_WEYL_VECTOR_B c102 += c64@q6, c2@q7; s2 x2 [run ends]
	vertex(&p[328], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[256], &p[116], NULL, NULL, &scales[4], &factors[4]);
	// 224: V_WEYL_VECTOR_B c103 = c31@q2, c37@q9; s2 x0 [run starts]
	vertex(&p[332], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[188], &p[220], NULL, NULL, &scales[4], &factors[0]);
	// 225: V_WEYL_VECTOR_B c103 += c62@q3, c7@q10; s2 x0 [in run]
	vertex(&p[332], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[296], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 226: V_WEYL_VECTOR_B c103 += c66@q6, c2@q7; s2 x2 [run ends]
	vertex(&p[332], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[260], &p[116], NULL, NULL, &scales[4], &factors[4]);
	// 227: V_WEYL_VECTOR_B c104 = c31@q2, c38@q9; s2 x0 [run starts]
	vertex(&p[336], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[188], &p[228], NULL, NULL, &scales[4], &factors[0]);
	// 228: V_WEYL_VECTOR_B c104 += c62@q3, c8@q10; s2 x0 [in run]
	vertex(&p[336], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[296], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 229: V_WEYL_VECTOR_B c104 += c67@q6, c2@q7; s2 x2 [run ends]
	vertex(&p[336], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[264], &p[116], NULL, NULL, &scales[4], &factors[4]);
	// 230: V_WEYL_VECTOR_A c113 = c75@q6, c4@q7; s2 x1 [run starts]
	vertex(&p[340], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[236], &p[124], NULL, NULL, &scales[4], &factors[2]);
	// 231: V_WEYL_VECTOR_A c113 += c29@q2, c35@q9; s2 x0 [in run]
	vertex(&p[340], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[172], &p[204], NULL, NULL, &scales[4], &factors[0]);
	// 232: V_WEYL_VECTOR_A c113 += c73@q3, c7@q10; s2 x0 [run ends]
	vertex(&p[340], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[300], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 233: V_WEYL_VECTOR_A c114 = c76@q6, c4@q7; s2 x1 [run starts]
	vertex(&p[344], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[240], &p[124], NULL, NULL, &scales[4], &factors[2]);
	// 234: V_WEYL_VECTOR_A c114 += c29@q2, c36@q9; s2 x0 [in run]
	vertex(&p[344], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[172], &p[212], NULL, NULL, &scales[4], &factors[0]);
	// 235: V_WEYL_VECTOR_A c114 += c73@q3, c8@q10; s2 x0 [run ends]
	vertex(&p[344], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[300], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 236: V_WEYL_VECTOR_A c115 = c77@q6, c4@q7; s2 x1 [run starts]
	vertex(&p[348], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[244], &p[124], NULL, NULL, &scales[4], &factors[2]);
	// 237: V_WEYL_VECTOR_A c115 += c29@q2, c37@q9; s2 x0 [in run]
	vertex(&p[348], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[172], &p[220], NULL, NULL, &scales[4], &factors[0]);
	// 238: V_WEYL_VECTOR_A c115 += c74@q3, c7@q10; s2 x0 [run ends]
	vertex(&p[348], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[304], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 239: V_WEYL_VECTOR_A c116 = c78@q6, c4@q7; s2 x1 [run starts]
	vertex(&p[352], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[248], &p[124], NULL, NULL, &scales[4], &factors[2]);
	// 240: V_WEYL_VECTOR_A c116 += c29@q2, c38@q9; s2 x0 [in run]
	vertex(&p[352], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[172], &p[228], NULL, NULL, &scales[4], &factors[0]);
	// 241: V_WEYL_VECTOR_A c116 += c74@q3, c8@q10; s2 x0 [run ends]
	vertex(&p[352], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[304], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 242: V_WEYL_VECTOR_B c105 = c32@q2, c35@q9; s2 x0 [run starts]
	vertex(&p[356], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[192], &p[204], NULL, NULL, &scales[4], &factors[0]);
	// 243: V_WEYL_VECTOR_B c105 += c65@q3, c7@q10; s2 x0 [in run]
	vertex(&p[356], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[308], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 244: V_WEYL_VECTOR_B c105 += c63@q6, c4@q7; s2 x2 [run ends]
	vertex(&p[356], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[252], &p[124], NULL, NULL, &scales[4], &factors[4]);
	// 245: V_WEYL_VECTOR_B c106 = c32@q2, c36@q9; s2 x0 [run starts]
	vertex(&p[360], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[192], &p[212], NULL, NULL, &scales[4], &factors[0]);
	// 246: V_WEYL_VECTOR_B c106 += c65@q3, c8@q10; s2 x0 [in run]
	vertex(&p[360], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[308], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 247: V_WEYL_VECTOR_B c106 += c64@q6, c4@q7; s2 x2 [run ends]
	vertex(&p[360], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[256], &p[124], NULL, NULL, &scales[4], &factors[4]);
	// 248: V_WEYL_VECTOR_B c107 = c32@q2, c37@q9; s2 x0 [run starts]
	vertex(&p[364], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[192], &p[220], NULL, NULL, &scales[4], &factors[0]);
	// 249: V_WEYL_VECTOR_B c107 += c68@q3, c7@q10; s2 x0 [in run]
	vertex(&p[364], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[312], &p[148], NULL, NULL, &scales[4], &factors[0]);
	// 250: V_WEYL_VECTOR_B c107 += c66@q6, c4@q7; s2 x2 [run ends]
	vertex(&p[364], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[260], &p[124], NULL, NULL, &scales[4], &factors[4]);
	// 251: V_WEYL_VECTOR_B c108 = c32@q2, c38@q9; s2 x0 [run starts]
	vertex(&p[368], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[192], &p[228], NULL, NULL, &scales[4], &factors[0]);
	// 252: V_WEYL_VECTOR_B c108 += c68@q3, c8@q10; s2 x0 [in run]
	vertex(&p[368], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[312], &p[156], NULL, NULL, &scales[4], &factors[0]);
	// 253: V_WEYL_VECTOR_B c108 += c67@q6, c4@q7; s2 x2 [run ends]
	vertex(&p[368], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[264], &p[124], NULL, NULL, &scales[4], &factors[4]);
	// 254: CLOSURE a0 += c19 . c117; list 1 x0
	closure(&p[44], &p[196], &p[92], cf1, 2, &factors[0]);
	// 255: CLOSURE a1 += c20 . c117; list 1 x0
	closure(&p[46], &p[200], &p[92], cf1, 2, &factors[0]);
	// 256: CLOSURE a2 += c81 . c117; list 1 x0
	closure(&p[48], &p[132], &p[92], cf1, 2, &factors[0]);
	// 257: CLOSURE a3 += c82 . c117; list 1 x0
	closure(&p[50], &p[136], &p[92], cf1, 2, &factors[0]);
	// 258: CLOSURE a4 += c87 . c117; list 1 x0
	closure(&p[52], &p[324], &p[92], cf1, 2, &factors[0]);
	// 259: CLOSURE a5 += c88 . c117; list 1 x0
	closure(&p[54], &p[328], &p[92], cf1, 2, &factors[0]);
	// 260: CLOSURE a6 += c89 . c117; list 1 x0
	closure(&p[56], &p[332], &p[92], cf1, 2, &factors[0]);
	// 261: CLOSURE a7 += c90 . c117; list 1 x0
	closure(&p[58], &p[336], &p[92], cf1, 2, &factors[0]);
	// 262: CLOSURE a8 += c95 . c117; list 1 x0
	closure(&p[60], &p[356], &p[92], cf1, 2, &factors[0]);
	// 263: CLOSURE a9 += c96 . c117; list 1 x0
	closure(&p[62], &p[360], &p[92], cf1, 2, &factors[0]);
	// 264: CLOSURE a10 += c97 . c117; list 1 x0
	closure(&p[64], &p[364], &p[92], cf1, 2, &factors[0]);
	// 265: CLOSURE a11 += c98 . c117; list 1 x0
	closure(&p[66], &p[368], &p[92], cf1, 2, &factors[0]);
	// 266: CLOSURE a12 += c118 . c79; list 1 x0
	closure(&p[68], &p[96], &p[100], cf1, 2, &factors[0]);
	// 267: CLOSURE a13 += c118 . c80; list 1 x0
	closure(&p[70], &p[96], &p[104], cf1, 2, &factors[0]);
	// 268: CLOSURE a14 += c118 . c14; list 1 x0
	closure(&p[72], &p[96], &p[176], cf1, 2, &factors[0]);
	// 269: CLOSURE a15 += c118 . c15; list 1 x0
	closure(&p[74], &p[96], &p[180], cf1, 2, &factors[0]);
	// 270: CLOSURE a16 += c118 . c83; list 1 x0
	closure(&p[76], &p[96], &p[140], cf1, 2, &factors[0]);
	// 271: CLOSURE a17 += c118 . c84; list 1 x0
	closure(&p[78], &p[96], &p[144], cf1, 2, &factors[0]);
	// 272: CLOSURE a18 += c118 . c85; list 1 x0
	closure(&p[80], &p[96], &p[316], cf1, 2, &factors[0]);
	// 273: CLOSURE a19 += c118 . c86; list 1 x0
	closure(&p[82], &p[96], &p[320], cf1, 2, &factors[0]);
	// 274: CLOSURE a20 += c118 . c91; list 1 x0
	closure(&p[84], &p[96], &p[340], cf1, 2, &factors[0]);
	// 275: CLOSURE a21 += c118 . c92; list 1 x0
	closure(&p[86], &p[96], &p[344], cf1, 2, &factors[0]);
	// 276: CLOSURE a22 += c118 . c93; list 1 x0
	closure(&p[88], &p[96], &p[348], cf1, 2, &factors[0]);
	// 277: CLOSURE a23 += c118 . c94; list 1 x0
	closure(&p[90], &p[96], &p[352], cf1, 2, &factors[0]);
	// 278: REDUCE out[12] += w1 × |a0 (× x0)|²
	out[12] = reduce_term(out[12], &p[44], &factors[0], 0.15639254553972046);
	// 279: REDUCE out[13] += w1 × |a1 (× x0)|²
	out[13] = reduce_term(out[13], &p[46], &factors[0], 0.15639254553972046);
	// 280: REDUCE out[14] += w1 × |a2 (× x0)|²
	out[14] = reduce_term(out[14], &p[48], &factors[0], 0.15639254553972046);
	// 281: REDUCE out[15] += w1 × |a3 (× x0)|²
	out[15] = reduce_term(out[15], &p[50], &factors[0], 0.15639254553972046);
	// 282: REDUCE out[16] += w1 × |a4 (× x0)|²
	out[16] = reduce_term(out[16], &p[52], &factors[0], 0.15639254553972046);
	// 283: REDUCE out[17] += w1 × |a5 (× x0)|²
	out[17] = reduce_term(out[17], &p[54], &factors[0], 0.15639254553972046);
	// 284: REDUCE out[18] += w1 × |a6 (× x0)|²
	out[18] = reduce_term(out[18], &p[56], &factors[0], 0.15639254553972046);
	// 285: REDUCE out[19] += w1 × |a7 (× x0)|²
	out[19] = reduce_term(out[19], &p[58], &factors[0], 0.15639254553972046);
	// 286: REDUCE out[20] += w1 × |a8 (× x0)|²
	out[20] = reduce_term(out[20], &p[60], &factors[0], 0.15639254553972046);
	// 287: REDUCE out[21] += w1 × |a9 (× x0)|²
	out[21] = reduce_term(out[21], &p[62], &factors[0], 0.15639254553972046);
	// 288: REDUCE out[22] += w1 × |a10 (× x0)|²
	out[22] = reduce_term(out[22], &p[64], &factors[0], 0.15639254553972046);
	// 289: REDUCE out[23] += w1 × |a11 (× x0)|²
	out[23] = reduce_term(out[23], &p[66], &factors[0], 0.15639254553972046);
	// 290: REDUCE out[24] += w1 × |a12 (× x0)|²
	out[24] = reduce_term(out[24], &p[68], &factors[0], 0.15639254553972046);
	// 291: REDUCE out[25] += w1 × |a13 (× x0)|²
	out[25] = reduce_term(out[25], &p[70], &factors[0], 0.15639254553972046);
	// 292: REDUCE out[26] += w1 × |a14 (× x0)|²
	out[26] = reduce_term(out[26], &p[72], &factors[0], 0.15639254553972046);
	// 293: REDUCE out[27] += w1 × |a15 (× x0)|²
	out[27] = reduce_term(out[27], &p[74], &factors[0], 0.15639254553972046);
	// 294: REDUCE out[28] += w1 × |a16 (× x0)|²
	out[28] = reduce_term(out[28], &p[76], &factors[0], 0.15639254553972046);
	// 295: REDUCE out[29] += w1 × |a17 (× x0)|²
	out[29] = reduce_term(out[29], &p[78], &factors[0], 0.15639254553972046);
	// 296: REDUCE out[30] += w1 × |a18 (× x0)|²
	out[30] = reduce_term(out[30], &p[80], &factors[0], 0.15639254553972046);
	// 297: REDUCE out[31] += w1 × |a19 (× x0)|²
	out[31] = reduce_term(out[31], &p[82], &factors[0], 0.15639254553972046);
	// 298: REDUCE out[32] += w1 × |a20 (× x0)|²
	out[32] = reduce_term(out[32], &p[84], &factors[0], 0.15639254553972046);
	// 299: REDUCE out[33] += w1 × |a21 (× x0)|²
	out[33] = reduce_term(out[33], &p[86], &factors[0], 0.15639254553972046);
	// 300: REDUCE out[34] += w1 × |a22 (× x0)|²
	out[34] = reduce_term(out[34], &p[88], &factors[0], 0.15639254553972046);
	// 301: REDUCE out[35] += w1 × |a23 (× x0)|²
	out[35] = reduce_term(out[35], &p[90], &factors[0], 0.15639254553972046);
	}

int run(double *out, double *p, const double *momenta) {
	for(int i=0; i<372; i++) p[i] = 0.0;
	for(int h=0; h<48; h++) out[h] = 0.0;
	forms(p, momenta);
	body(out, p);
	return 0;
	}
