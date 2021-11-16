#### C++ML
In progress. In this project I experiment with some stuff. So the code is not optimised and it does not run on GPU.

Implements matrix maths with LU decomposition and Linear Programming. Can be used for classification (linear, polynomial, Fouirer).

Matrix.h
```
  // create matrix
  Matrix<float> A(3, 3, 1);
  A.print();
  Matrix<float> B("1 0 0, 0 1 0, 0 0 1"); // identity
  B.print();
  
  // matrix from file (first line must be like "1 0 0, 0 1 0, 0 0 1")
  // Matrix<float> C("filename.mat");
  
  // save matrix
  A.save("mat.mat");
  
  // addition and scalar multiplication
  (A+B).print();
  (B*(-1)).print();
  
  // transpose
  A.transpose().print();
  
  // apply function to all entries
  Matrix<float> vec_in(4, 1, 0);
  Matrix<float> weights(2, 4, 0);
  auto vec_out = weights*vec_in;
  vec_out.applyFunction([](float a) { return (a >= 0) ? 1 : 0; } );
  
  // add row or column
  B.addRow(vector<float>{1, 1, 1});
  B.print();
  B.addColumn(vector<float>{1,1,1,1});
  B.print();
  
  // access entry or get dimension
  A.data[0][0]; // first row first column
  A.iRow;       // number rows
  A.iColumn;    // number columns
```
  
matrix_utils.h
```
  /*
    The linear program
			max 3 x1 + 5 x2 + 0
			  x1 		<= 4
				   2 x2 <= 12
			3 x1 + 2 x2 <= 18
		corresponds to matrix
			-3 -5 0
			 1  0 4
			 0  2 12
			 3  2 18
  */
  //
  Matrix<float> A("-3 -5 0, 1 0 4, 0 2 12, 3 2 18");
  A.print();
	
  // solve LP to final Tableau
  auto A_solution = LPsolver(A);
  A_solution.print();
  
  // solve LP to variables
  solveLP_variables(Matrix<float>("0 0 0, 0 -1 1, 1 1 1")).print();
  
  // LU Decomposition
  Matrix<float> A("1 0 1, 0 1 0, 0 0 1");
  auto LU = LUdecomposition(A);
  LU.first.print();
  LU.second.print();
```
