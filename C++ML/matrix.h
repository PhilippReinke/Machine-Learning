#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <type_traits>
#include <functional>
#include <sstream>
#include <fstream>

using namespace std;

// Hilfsfunktionen
vector<string> split_string(string str, char seperator)
{
	vector<string> result;
	stringstream s_stream(str);
	while(s_stream.good()) 
	{
		string substr;
		getline(s_stream, substr, seperator);
		result.push_back(substr);
	}
	return result;
}

template <class T> 
class Matrix
{
public:
	// Variablen
	unsigned iRow=0, iColumn=0;
	vector<vector<T>> data;

	// Konstruktoren
	Matrix(unsigned row, unsigned col)
	{
		iRow = row; iColumn = col;
		data = vector<vector<T>>(row, vector<T>(col, (T) 0));
	}
	Matrix(unsigned row, unsigned col, T value)
	{
		iRow = row; iColumn = col;
		data = vector<vector<T>>(row, vector<T>(col, value));
	}
	Matrix(string strMatrix) //  strMatrix = "1 1 1, 2 2 2, 3 3 3" oder Dateiname
	{
		// Dateiname oder String?
		bool bContainsLetter = false;
		for (auto &ele : strMatrix)
		{
			auto i = tolower(int(ele));
			if (int('a') <= i &&  int('z') >= i)
			{
				bContainsLetter = true;
				break;
			}
		}

		// erste Zeile aus Datei lesen
		if (bContainsLetter)
		{
			string line;
			ifstream file(strMatrix);
			if (file.is_open())
			{
				getline(file, line);
				file.close();
			}
			else 
				cout << "Error: Datei kann nicht geöffnet werden." << endl;

			// im nächste Schritt werden Daten ausgelesen
			strMatrix = line;
		}

		// Daten aus dem String lesen
		try
		{
			vector<string> rows = split_string(strMatrix, ',');
			iRow = (unsigned)rows.size();
			for(string &row : rows)
			{
				vector<T> rowNumerical;
				vector<string> entries = split_string(row, ' ');
				for(string &entry : entries)
				{
					if(entry == "")
						continue;
					if(entry.find('.') != std::string::npos)
						rowNumerical.push_back((T) stold(entry));
					else
						rowNumerical.push_back((T) stoi(entry));
				}
				data.push_back(rowNumerical);
				if(iColumn == 0)
					iColumn = (unsigned)rowNumerical.size();
			}
		}
		catch (...) { cout << "Error: Matrix kann nicht gelesen werden."; }
	}
	Matrix(vector<vector<T>> dataMat)
	{
		data = dataMat;
		iRow = (unsigned)dataMat.size();
		iColumn = (unsigned)dataMat[0].size();
	}

	// Infos
	void print()
	{
		for(auto &row : data)
		{
			for(auto &entry : row)
				cout << entry << " ";
			cout << endl;
		}
		cout << endl;
	}
	void printType() { cout << typeid(T).name() << endl; }

	// Matrix als Datei speichern
	void save(string filename)
	{
		// Matrix zu String
		string mat_string;
		for (auto &row : data)
		{
			for (auto &ele : row)
			{
				string temp = to_string(ele);
				temp.erase( temp.find_last_not_of('0') + 1, string::npos );
				mat_string +=  temp + ' ';
			}
			if (&row != &data.back())
				mat_string += ',';
		}

		// Matrix in Datei speichern
		ofstream file_out;
		file_out.open (filename);
		file_out << mat_string;
		file_out.close();
	}

	// Matrix transponieren
	Matrix<T> transpose()
	{
		Matrix<T> mat(iColumn, iRow);
		for(unsigned m=0; m<iRow; m++)
		{
			for(unsigned n=0; n<iColumn; n++)
				mat.data[n][m] = data[m][n];
		}
		return mat;
	}

	// Zeile oder Spalte hinzufügen
	void addRow(vector<T> vec)
	{
		if(vec.size() != iColumn)
			cerr << "Error: Matrix hat " << iColumn << " Spalten und nicht "<< vec.size() << "." << endl;
		else
		{
			data.push_back(vec);
			iRow++;
		}
	}
	void addColumn(vector<T> vec)
	{
		if(vec.size() != iRow)
			cerr << "Error: Matrix hat " << iRow << " Zeilen und nicht "<< vec.size() << "." << endl;
		else
		{
			for(unsigned i=0; i<vec.size(); i++)
				data[i].push_back(vec[i]);
			iColumn++;
		}
	}

	// Matrixaddition
	template<typename U>
	auto operator+(const Matrix<U> &matA)
	{
		// "kleinsten gemeinsamen" Datentyp bestimmen
		auto Zero = (decltype(data[0][0]+matA.data[0][0])) 0;
		//auto x = (make_signed<decltype(Zero)>) 0;
		Matrix<decltype(Zero)> matB(iRow, iColumn, Zero);

		// Matrizen addieren
		if(iRow == matA.iRow && iColumn == matA.iColumn)
		{
			for(unsigned m=0; m<iRow; m++)
			{
				for(unsigned n=0; n<iColumn; n++)
				{
					matB.data[m][n] = data[m][n] + matA.data[m][n];
				}
			}
		}
		else
		{ 
			cerr << "Error: Matrizen unterschiedlicher Dimension können nicht addiert werden." << endl;
		}
		return matB;
	}

	// (Lambda-)Funktion auf alle Einträge anwenden
	template<typename U>
	void applyFunction(U func)
	{
		for(auto &row : data)
			for(auto &entry : row)
				entry = (T) func(entry);
	}

	// Matrixmultiplikation
	template<typename U>
	auto operator*(const Matrix<U> &matA)
	{
		// "kleinsten gemeinsamen" Datentyp bestimmen
		auto Zero = (decltype(data[0][0]+matA.data[0][0])) 0;
		Matrix<decltype(Zero)> matB(iRow, matA.iColumn, Zero);

		// Matrizen multiplizieren
		if(iColumn == matA.iRow)
		{
			for(unsigned m=0; m<iRow; m++)
			{
				for(unsigned n=0; n<matA.iColumn; n++)
				{
					T sum = 0;
					for(unsigned o=0; o<matA.iRow; o++)
						sum += data[m][o] * matA.data[o][n];
					matB.data[m][n] = sum;
				}
			}
		}
		else
		{ 
			cerr << "Error: Dimensionen erlauben keine Multiplikation." << endl;
		}
		return matB;
	}

	// Skalarmultiplikation
	template<typename U>
	auto operator*(const U &scalar)
	{
		// "kleinsten gemeinsamen" Datentyp bestimmen
		auto Zero = (decltype(data[0][0]+scalar)) 0;
		Matrix<decltype(Zero)> mat(iRow, iColumn, Zero);

		// multipliziere Einträge mit Skalar
		for(unsigned m=0; m<iRow; m++)
		{
			for(unsigned n=0; n<iColumn; n++)
			{
				mat.data[m][n] = data[m][n]*scalar;
			}
		}
		return mat;
	}
};

#endif

/*

= = = Beispiel

// Addition
Matrix<float> A(3, 3, 1);

// Matrix aus String
Matrix<int> B("1 0 0, 0 1 0, 0 0 1");
auto Bt = B.transpose();
Bt.print();
Matrix<float> C("1 0 0, 0 1 0, 0 0 1, 0 2.2 0, 0 0 0");

// Multiplikation
(A+B).print();
(A*B).print();
(C*B).print();

// Skalarmultiplikation
(B*2).print();

// Funktion auf Einträge anwenden
B.applyFunction( [](int a) { return a+2; } );
B.print();

Matrix<float> vec_in(4, 1, 0);
Matrix<float> weights(2, 4, 0);
auto vec_out = weights*vec_in;
vec_out.applyFunction([](float a) { return (a >= 0) ? 1 : 0; } );
vec_out.print();

Matrix<float> mat("1 0 0, 0 1 0, 0 0 1");
mat.print();
mat.addRow(vector<float>{1,1,1});
mat.print();
mat.addColumn(vector<float>{1,1,1,1});
mat.print();

*/