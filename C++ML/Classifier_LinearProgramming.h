#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "matrix.h"
#include "matrix_utils.h"

// Prototypen
pair<unsigned, unsigned> evaluate_prediction(vector<Matrix<double>> &y_data, vector<Matrix<double>> &y_pred);
int check_data(vector<Matrix<double>> &x_data, vector<Matrix<double>> &y_data);

vector<Matrix<double>> Polynomial_PrepareX(vector<Matrix<double>> &x_data, unsigned iDegree);
Matrix<double> PolynomialClassification(vector<Matrix<double>> &x_data, vector<Matrix<double>> &y_data, unsigned iDegree);			// trainieren
vector<Matrix<double>> PolynomialClassification(Matrix<double> &weightMatrix, vector<Matrix<double>> &x_data, unsigned iDegree);	// vorhersagen

vector<Matrix<double>> Fourier_PrepareX(vector<Matrix<double>> &x_data);
Matrix<double> FourierClassification(vector<Matrix<double>> &x_data, vector<Matrix<double>> &y_data, unsigned iDegree); 			// trainieren
vector<Matrix<double>> FourierClassification(Matrix<double> &weightMatrix, vector<Matrix<double>> &x_data, unsigned iDegree); 		// vorhersagen

Matrix<double> LinearClassification_LP(vector<Matrix<double>> &x_data, vector<Matrix<double>> &y_data);

// alle
pair<unsigned, unsigned> evaluate_prediction(vector<Matrix<double>> &y_data, vector<Matrix<double>> &y_pred)
{
	unsigned iCorrect = 0;
	for (unsigned i=0; i<y_data.size(); i++)
		if (y_data[i].data == y_pred[i].data)
			iCorrect++;
	return make_pair(iCorrect, y_data.size());
}
int check_data(vector<Matrix<double>> &x_data, vector<Matrix<double>> &y_data)
{
	if (x_data.size() != y_data.size())
	{
		cerr << "Error: Datensätze enthalten unterschiedlich viele Matrizen." << endl;
		return 1;
	}
	for (unsigned i=0; i<x_data.size(); i++)
	{
		if (x_data[i].iRow != x_data[0].iRow)
		{
			cerr << "Error in x: Zeilenanzahl unterschiedlich für Index " << x_data[i].iRow << endl;
			return 1;
		}
		if (x_data[i].iColumn != x_data[0].iColumn)
		{
			cerr << "Error in x: Spaltenanzahl unterschiedlich für Index " << x_data[i].iRow << endl;
			return 1;
		}
		if (y_data[i].iRow != y_data[0].iRow)
		{
			cerr << "Error in y: Zeilenanzahl unterschiedlich für Index " << x_data[i].iRow << endl;
			return 1;
		}
		if (y_data[i].iColumn != y_data[0].iColumn)
		{
			cerr << "Error in y: Spaltenanzahl unterschiedlich für Index " << x_data[i].iRow << endl;
			return 1;
		}
	}
	return 0;
}

// POLYNOMIAL
vector<Matrix<double>> Polynomial_PrepareX(vector<Matrix<double>> &x_data, unsigned iDegree)
{
	unsigned x_num_rows= x_data[0].iRow;
	vector<Matrix<double>> x_copy = x_data;

	// x^2, x^3, ... anfügen
	for (auto &ele : x_copy)
	{
		vector<vector<double>> vec = ele.data;
		for (unsigned i=2; i<=iDegree; i++)
		{
			for (unsigned j=0; j<x_num_rows; j++)
				vec[j][0] *= ele.data[j][0];
			ele.data.insert( ele.data.end(), vec.begin(), vec.end() );
			ele.iRow += vec.size();
		}
	}

	// x_data[0].print();
	// x_copy[0].print();
	// cout << "perpareX_END" << endl;

	return x_copy;
}
Matrix<double> PolynomialClassification(vector<Matrix<double>> &x_data, vector<Matrix<double>> &y_data, unsigned iDegree) // trainieren
{
	vector<Matrix<double>> x_copy = Polynomial_PrepareX(x_data, iDegree);
	return LinearClassification_LP(x_copy, y_data);
}
vector<Matrix<double>> PolynomialClassification(Matrix<double> &weightMatrix, vector<Matrix<double>> &x_data, unsigned iDegree) // vorhersagen
{
	vector<Matrix<double>> y_predictions;
	vector<Matrix<double>> x_data_new = Polynomial_PrepareX(x_data, iDegree);

	for (auto &ele : x_data_new)
	{
		ele.addRow(vector<double>{1});
		auto vec_pred  = weightMatrix*ele;

		vec_pred.applyFunction([](double x) { return (x >= 0) ? 1 : -1; } );
		y_predictions.push_back(vec_pred);
	}
	return y_predictions;
}

// Fourier
vector<Matrix<double>> Fourier_PrepareX(vector<Matrix<double>> &x_data, unsigned iDegree)
{
	// (x1, x2, x3) geht nach
	// (cos(x1), sin(x1), cos(2*x1), sin(2*x1), cos(3*x1), sin(3*x1),..., cos(x2), sin(x2), ...)
	vector<Matrix<double>> x_data_new;

	for (auto &mat : x_data)
	{
		vector<vector<double>> newMat;
		for (auto &ele : mat.data)
		{	
			for (unsigned k=1; k<=iDegree; k++)
			{
				newMat.push_back(vector<double>{cos(k*ele[0])});
				newMat.push_back(vector<double>{sin(k*ele[0])});
			}
		}
		x_data_new.push_back(Matrix<double>(newMat));
	}
	return x_data_new;
}
Matrix<double> FourierClassification(vector<Matrix<double>> &x_data, vector<Matrix<double>> &y_data, unsigned iDegree) // trainieren
{
	vector<Matrix<double>> x_data_new = Fourier_PrepareX(x_data, iDegree);
	return LinearClassification_LP(x_data_new, y_data);
}
vector<Matrix<double>> FourierClassification(Matrix<double> &weightMatrix, vector<Matrix<double>> &x_data, unsigned iDegree) // vorhersagen
{
	vector<Matrix<double>> y_predictions;
	vector<Matrix<double>> x_data_new = Fourier_PrepareX(x_data, iDegree);

	for (auto &ele : x_data_new)
	{
		ele.addRow(vector<double>{1});
		auto vec_pred  = weightMatrix*ele;

		vec_pred.applyFunction([](double x) { return (x >= 0) ? 1 : -1; } );
		y_predictions.push_back(vec_pred);
	}

	return y_predictions;
}

// Linear
Matrix<double> LinearClassification_LP(vector<Matrix<double>> &x_data, vector<Matrix<double>> &y_data)
{
	Matrix<double> weights(0, x_data[0].iRow+1, 0);

	for (unsigned y_row=0; y_row<y_data[0].iRow; y_row++)
	{
		// Matrix für LP bauen (erste Zeile ist Zielfunktion mit Nullen)
		Matrix<double> mat((unsigned)x_data.size()+1, x_data[0].iRow+1, 0);
		// Werte einsetzen
		for (unsigned i=1; i<x_data.size()+1; i++)
		{
			double &y = y_data[i-1].data[y_row][0];
			for (unsigned k=0; k<x_data[0].iRow; k++)
				mat.data[i][k] = y*x_data[i-1].data[k][0];
			mat.data[i][x_data[0].iRow] = y;
		}
		// rechte Seite hinzufügen
		vector<double> rightSide((unsigned)x_data.size()+1, 1);
		rightSide[0] = 0;
		mat.addColumn(rightSide);

		// löse LP
		auto matSolution = solveLP_variables(mat);
		
		// Gewichte speichern
		weights.addRow(matSolution.data[0]);

		// Status ausgeben
		//cout << y_row+1 << " / " << y_data[0].iRow << " geschafft." << endl;
	}

	return weights;
}

#endif