
__kernel void vmultiply(__global float* result, __global const float* m_a, __global const float* m_b,
						const int P, const int M, const int N){
	/*
		m_a ia a PxM matrix
		m_b is a MxN matrix
		result = m_b * m_b
	*/
	// Get global position in X direction
	int col = get_global_id(0);
	// Get global position in Y direction
	int row = get_global_id(1);

	// calculate the result of one element result[row][col]
	float sum = 0.0f;
	if((row>=0) && (row<P) && (col>=0) && (col<N)){ 
		for(int i=0; i<M; ++i){ 
			sum += m_a[row*M+i] * m_b[i*N+col];
		}
	}
	result[row*N+col] = sum;
}
