#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "matrix.h"

// Prototypen
Matrix<float> LPsolver(Matrix<float>);
Matrix<float> LinearClassification_LP(vector<Matrix<float>> &x_data, vector<Matrix<float>> &y_data);
Matrix<float> solveLP_variables(Matrix<float> &mat);
void check_data();

vector<Matrix<float>> Polynomial_PrepareX(vector<Matrix<float>> &x_data, unsigned iDegree);
Matrix<float> PolynomialClassification(vector<Matrix<float>> &x_data, vector<Matrix<float>> &y_data, unsigned iDegree);			// trainieren
vector<Matrix<float>> PolynomialClassification(Matrix<float> &weightMatrix, vector<Matrix<float>> &x_data, unsigned iDegree);	// vorhersagen

vector<Matrix<float>> Fourier_PrepareX(vector<Matrix<float>> &x_data);
Matrix<float> FourierClassification(vector<Matrix<float>> &x_data, vector<Matrix<float>> &y_data, unsigned iDegree); 			// trainieren
vector<Matrix<float>> FourierClassification(Matrix<float> &weightMatrix, vector<Matrix<float>> &x_data, unsigned iDegree); 		// vorhersagen

// ALL
pair<unsigned, unsigned> evaluate_prediction(vector<Matrix<float>> &y_data, vector<Matrix<float>> &y_pred)
{
	unsigned iCorrect = 0;
	for (unsigned i=0; i<y_data.size(); i++)
		if (y_data[i].data == y_pred[i].data)
			iCorrect++;
	return make_pair(iCorrect, y_data.size());
}
void check_data()
{
	cout << "Alles ok." << endl;
}
Matrix<float> solveLP_variables(Matrix<float> &mat)
{
/*
	Löst LP und rechnet Variablen ohne Schlupfvariablen aus.
	Beispiel:
	solveLP_variables(Matrix<float>("0 0 0, 0 -1 1, 1 1 1")).print();
*/
	// löse LP
	auto matSolution = LPsolver(mat);

	// neue Matrix bestehend aus Variablen ohne Schlupfvariablen
	unsigned num_variables = mat.iColumn-1;
	unsigned num_samples   = mat.iRow-1;
	Matrix<float> mat_clean(num_samples, num_variables, 0);
	for (unsigned m=0; m<mat_clean.iRow; m++)
		for (unsigned n=0; n<mat_clean.iColumn; n++)
			mat_clean.data[m][n] = matSolution.data[m+1][n];

	// Normalengleichung
	auto normal_matrix 	= mat_clean.transpose()*mat_clean;
	auto right_side 	= mat_clean.transpose()*Matrix<float>(num_samples, 1, 1);

	// LR-Zerlegung
	unsigned x_size = normal_matrix.iRow;
	Matrix<float> L(x_size, x_size, 0);
	Matrix<float> U(x_size, x_size, 0);
	auto &u = U.data, &l = L.data, &a = normal_matrix.data;
	for(unsigned i = 0; i<x_size; i++)
	{
		for(unsigned j=0; j<x_size; j++) 
		{
			if (j < i)
				l[j][i] = 0;
			else 
			{
				l[j][i] = a[j][i];
				for(unsigned k=0; k<i; k++)
					l[j][i] = l[j][i] - l[j][k] * u[k][i];
			}
			// ToDo: gefährlich
			if (isnan(l[i][j]))
				l[i][j] = 0;
		}
		for(unsigned j=0; j<x_size; j++) 
		{
			if (j < i)
				u[i][j] = 0;
			else if (j == i)
				u[i][j] = 1;
			else
			{
				u[i][j] = a[i][j] / l[i][i];
				for (unsigned k=0; k<i; k++)
					u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
			}
			// ToDo: gefährlich
			if (isnan(u[i][j]))
				u[i][j] = (float)0;
		}
	}

	// Nullen auf Diagonale auffüllen, falls es keine eindeutige Lösung gibt
	for (unsigned i=0; i<x_size; i++)
	{
		if (l[i][i] == (float)0)
			l[i][i] = (float)1;
		if (u[i][i] == (float)0)
			u[i][i] = (float)1;
	}

	// forward substitution
	vector<float> v(x_size); auto &b = right_side.data;
	for(unsigned i=0; i<x_size; i++)
	{
		v[i] = b[i][0];
		for(unsigned j=0; j<i; j++)
			v[i] -= l[i][j]*v[j];
		v[i] = v[i] / l[i][i];
	}
	// backward substitution
	vector<float> w(x_size);
	for(unsigned i=x_size-1; i+1>=1; i--)
	{
		w[i] = v[i];
		for(unsigned j=i+1; j<x_size; j++)
		{
			w[i] -= u[i][j]*w[j];
		}
		w[i] = w[i] / u[i][i];
	}

	return Matrix<float>(vector<vector<float>>{w});
}

// FOURIER
vector<Matrix<float>> Fourier_PrepareX(vector<Matrix<float>> &x_data, unsigned iDegree)
{
	// (x1, x2, x3) geht nach
	// (cos(x1), sin(x1), cos(2*x1), sin(2*x1), cos(3*x1), sin(3*x1),..., cos(x2), sin(x2), ...)
	vector<Matrix<float>> x_data_new;

	for (auto &mat : x_data)
	{
		vector<vector<float>> newMat;
		for (auto &ele : mat.data)
		{	
			for (unsigned k=1; k<=iDegree; k++)
			{
				newMat.push_back(vector<float>{cos(k*ele[0])});
				newMat.push_back(vector<float>{sin(k*ele[0])});
			}
		}
		x_data_new.push_back(Matrix<float>(newMat));
	}

	return x_data_new;
}
Matrix<float> FourierClassification(vector<Matrix<float>> &x_data, vector<Matrix<float>> &y_data, unsigned iDegree) // trainieren
{
	vector<Matrix<float>> x_data_new = Fourier_PrepareX(x_data, iDegree);
	return LinearClassification_LP(x_data_new, y_data);
}
vector<Matrix<float>> FourierClassification(Matrix<float> &weightMatrix, vector<Matrix<float>> &x_data, unsigned iDegree) // vorhersagen
{
	vector<Matrix<float>> y_predictions;
	vector<Matrix<float>> x_data_new = Fourier_PrepareX(x_data, iDegree);

	for (auto &ele : x_data_new)
	{
		ele.addRow(vector<float>{1});
		auto vec_pred  = weightMatrix*ele;

		vec_pred.applyFunction([](float x) { return (x >= 0) ? 1 : -1; } );
		y_predictions.push_back(vec_pred);
	}

	return y_predictions;
}

// POLYNOMIAL
vector<Matrix<float>> Polynomial_PrepareX(vector<Matrix<float>> &x_data, unsigned iDegree)
{
	unsigned x_num_rows= x_data[0].iRow;
	vector<Matrix<float>> x_copy = x_data;

	// x^2, x^3, ... anfügen
	for (auto &ele : x_copy)
	{
		vector<vector<float>> vec = ele.data;
		for (unsigned i=2; i<=iDegree; i++)
		{
			for (unsigned j=0; j<x_num_rows; j++)
				vec[j][0] *= ele.data[j][0];
			ele.data.insert( ele.data.end(), vec.begin(), vec.end() );
			ele.iRow += vec.size();
		}
	}

	return x_copy;
}
Matrix<float> PolynomialClassification(vector<Matrix<float>> &x_data, vector<Matrix<float>> &y_data, unsigned iDegree) // trainieren
{
	vector<Matrix<float>> x_copy = Polynomial_PrepareX(x_data, iDegree);
	return LinearClassification_LP(x_copy, y_data);
}
vector<Matrix<float>> PolynomialClassification(Matrix<float> &weightMatrix, vector<Matrix<float>> &x_data, unsigned iDegree) // vorhersagen
{
	vector<Matrix<float>> y_predictions;
	vector<Matrix<float>> x_data_new = Polynomial_PrepareX(x_data, iDegree);

	for (auto &ele : x_data_new)
	{
		ele.addRow(vector<float>{1});
		auto vec_pred  = weightMatrix*ele;

		vec_pred.applyFunction([](float x) { return (x >= 0) ? 1 : -1; } );
		y_predictions.push_back(vec_pred);
	}
	return y_predictions;
}

// LINEAR
Matrix<float> LinearClassification_LP(vector<Matrix<float>> &x_data, vector<Matrix<float>> &y_data)
{
	Matrix<float> weights(0, x_data[0].iRow+1, 0);

	for (unsigned y_row=0; y_row<y_data[0].iRow; y_row++)
	{
		// Matrix für LP bauen (erste Zeile ist Zielfunktion mit Nullen)
		Matrix<float> mat((unsigned)x_data.size()+1, x_data[0].iRow+1, 0);
		// Werte einsetzen
		for (unsigned i=1; i<x_data.size()+1; i++)
		{
			float &y = y_data[i-1].data[y_row][0];
			for (unsigned k=0; k<x_data[0].iRow; k++)
				mat.data[i][k] = y*x_data[i-1].data[k][0];
			mat.data[i][x_data[0].iRow] = y;
		}
		// rechte Seite hinzufügen
		vector<float> rightSide((unsigned)x_data.size()+1, 1);
		rightSide[0] = 0;
		mat.addColumn(rightSide);

		// löse LP
		auto matSolution = solveLP_variables(mat);
		
		// Gewichte speichern
		weights.addRow(matSolution.data[0]);

		// Status ausgeben
		// cout << y_row+1 << " / " << y_data[0].iRow << " geschafft." << endl;
	}

	return weights;
}

Matrix<float> LPsolver(Matrix<float> mat)
{
/*
	= = = = = = = = = = = = = =
	= Beispiel zum LP Solver  =
	= = = = = = = = = = = = = =

		Das lineare Programm
			max 3 x1 + 5 x2 + 0
			  x1 		<= 4
				   2 x2 <= 12
			3 x1 + 2 x2 <= 18
		wird zur Matrix
			-3 -5 0
			 1  0 4
			 0  2 12
			 3  2 18
		und hat die Lösung
			0 0 0 1.5 		1 		  36
			0 0 1 0.333333  -0.333333 2
			0 1 0 0.5 		0 		  6
			1 0 0 -0.333333 0.333333  2
		Also
			Maximum: 36
			x1 = 2
			x2 = 6

		= = = = Code = = = =

		// lineares Programm eingeben
		Matrix<float> A("-3 -5 0, 1 0 4, 0 2 12, 3 2 18");
		A.print();

		// löse LP
		auto A_solution = LPsolver(A);
		A_solution.print();

		// Werte der Variablen ausrechnen
		Matrix<float> weights(A.iColumn-1, 1, 0);
		for (unsigned i=0; i<A.iColumn-1; i++)
		{
			for (unsigned j=1; j<A_solution.iRow; j++)
			{
				if (A_solution.data[j][i] != 0.0)
				{
					weights.data[i][0] = A_solution.data[j].back() / A_solution.data[j][i];
					break;
				}
			}
		}
		weights.print();
*/

	unsigned n = mat.iColumn-1;	// Anzahl der Variablen
	unsigned k = mat.iRow-1;	// Anzahl der Nebenbedingungen
	auto &mat_raw = mat.data;

	// Kontrolliere, ob negative Werte bei b vorhanden sind.
	bool bcontrol = false;
	float bvektor[k+1];

	for(unsigned i=0; i<(k+1); i++)
	{
		if(mat_raw[i][n] < 0)
			bcontrol=true;
		bvektor[i]=mat_raw[i][n];
	}
	
	if(bcontrol == true)
	{
		cerr << "negative Rechte-Seite, dieser Spezialfall wird hier nicht behandelt!" << endl;
		return Matrix<float>(0,0,0);
	}

	//
	Matrix<float> mat_out(k+1, n+k+1);
	auto &matrix = mat_out.data;

	for(unsigned i=0; i<(k+1); i++)
		for(unsigned j=0; j < (n+k+1);j++)
			matrix[i][j]=mat_raw[i][j];

	// mit 0 auffüllen
	for(unsigned i=0; i<(k+1); i++)
		for(unsigned j=n; j<(n+k+1); j++)
			matrix[i][j]=0;

	// Schlupfvariablen einfügen
	for(unsigned i=1; i<(k+1); i++)
		for(unsigned j=n; j<(n+k+1); j++)
			matrix[i][n-1+i]=1;

	// Vektor b einfügen
	for(unsigned i=0; i<(k+1); i++)
		matrix[i][n+k]=bvektor[i];

	for (int zaehler=0; zaehler<(int)(n+k); zaehler++) 
	{
		if (matrix[0][(unsigned)zaehler]<0)
		{
			// Pivotspalte suchen, Minimum in Zielfunktion
			float pivots = matrix[0][0];
			unsigned indexs = 0;
			for (unsigned i=1; i<(n+k); i++) 
			{
				if(matrix[0][i]<0) 
				{
					if (matrix[0][i]<pivots) 
					{
						pivots=matrix[0][i];
						indexs=i;
					}
				}
			}

			// erstes Element in der Pivotspalte ungleich 0
			float positiv = matrix[indexs][1];
			unsigned indexz = 1;
			for (unsigned j=1; j<k; j++) 
			{
				if(matrix[j][indexs]>0) 
				{
					positiv = matrix [j][indexs];
					indexz=j;
					break;
				}
			}

			if (positiv<=0)
			{
				cerr << "das lineare Programm ist unbeschränkt! --> Unendliche Lösung!"<< endl;
				return Matrix<float>(0,0,0);
			}

			// Pivotelement suchen,
			float pivotsuche=(matrix[indexz][n+k])/(matrix[indexz][indexs]);
			float pivot=matrix[indexz][indexs];
			for (unsigned i=indexz; i<(k+1); i++) 
			{
				if (matrix[i][indexs]>0) 
				{
					if (matrix[i][n+k]/matrix[i][indexs]<pivotsuche) 
					{
						pivotsuche=matrix[i][n+k]/matrix[i][indexs];
						indexz=i;
						pivot=matrix[indexz][indexs];
					}
				}
			}

			for (unsigned i=0; i<(k+1); i++) 
			{
				float hilf=matrix[i][indexs];
				for(unsigned j=0; j<(n+k+1); j++) 
				{
					if (i != indexz) 
					{
						if (j != indexs)
							matrix[i][j]=matrix[i][j]-((hilf/pivot)*matrix[indexz][j]);
						else 
							matrix[i][j]=0;
					}
				}
			}

			// Pivotzeile verändern
			for (unsigned i=0; i<n+k+1; i++)
				matrix[indexz][i]=matrix[indexz][i]/pivot;

			zaehler = -1;
		}
	}

	return mat_out;
}

#endif