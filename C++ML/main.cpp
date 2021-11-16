#include <algorithm>
#include <random>

#include "Classifier_LinearProgramming.h"

// Prototypen
vector<vector<Matrix<float>>> load_Iris(float train_test_split);
vector<vector<Matrix<float>>> load_MNIST(float train_test_split, int num_examples);

int main()
{
	// Iris laden
	auto iris_data = load_Iris(0.2);
	auto &x_train = iris_data[0]; 
	auto &y_train = iris_data[1];
	auto &x_test  = iris_data[2];
	auto &y_test  = iris_data[3];
	check_data(x_train, y_train); check_data(x_test,  y_test);

	// Lineare Klassifizierung
	auto w_lin 		= PolynomialClassification(x_train, y_train, 1);
	auto pred_lin 	= PolynomialClassification(w_lin, x_train, 1);
	auto res_lin 	= evaluate_prediction(y_train, pred_lin);
	auto pred_lin_test 	= PolynomialClassification(w_lin, x_test, 1);
	auto res_lin_test 	= evaluate_prediction(y_test, pred_lin_test);
	cout << "Lineare Klassifizierung     : Performance " << res_lin_test.first / (float) res_lin_test.second << endl;

	// Polynomiale Klassifizierung
	unsigned MAX_POLY_DEGREE = 10; // >= 2
	float best_ratio_poly = -1.0; unsigned best_degree_poly;
	for (unsigned degree=2; degree<=MAX_POLY_DEGREE; degree++)
	{
		auto w 		= PolynomialClassification(x_train, y_train, degree);
		auto pred 	= PolynomialClassification(w, x_train, degree);
		auto res 	= evaluate_prediction(y_train, pred);
		auto pred_test 	= PolynomialClassification(w, x_test, degree);
		auto res_test 	= evaluate_prediction(y_test, pred_test);
		float current_ratio = res_test.first / (float) res_test.second;
		if (current_ratio > best_ratio_poly) { best_ratio_poly = current_ratio; best_degree_poly = degree; }
	}
	cout << "Polynomiale Klassifizierung : Performance " << best_ratio_poly << " mit Grad " << best_degree_poly << endl;

	// Fourier Klassifizierung
	unsigned MAX_FOURIER_DEGREE = 10; // >= 1
	float best_ratio_fourier = -1.0; unsigned best_degree_fourier;
	for (unsigned degree=1; degree<=MAX_FOURIER_DEGREE; degree++)
	{
		auto w 		= FourierClassification(x_train, y_train, degree);
		auto pred 	= FourierClassification(w, x_train, degree);
		auto res 	= evaluate_prediction(y_train, pred);
		auto pred_test 	= FourierClassification(w, x_test, degree);
		auto res_test 	= evaluate_prediction(y_test, pred_test);
		float current_ratio = res_test.first / (float) res_test.second;
		if (current_ratio > best_ratio_fourier) { best_ratio_fourier = current_ratio; best_degree_fourier = degree; }
	}
	cout << "Fourier Klassifizierung     : Performance " << best_ratio_fourier << " mit Grad " << best_degree_fourier << endl;

	return 0;
}

vector<vector<Matrix<float>>> load_Iris(float train_test_split)
{
	vector<Matrix<float>> x, y;
	ifstream file_train("datasets/iris.csv");
	if (file_train.is_open())
	{
		string line;
		while ( getline(file_train, line) )
		{
			// y nach one-hot-encoded Vektor (mit 1=Treffer, -1=Nein)
			unsigned y_num  = line.back() - '0';
			Matrix<float> y_matrix(3, 1, -1); y_matrix.data[y_num-1][0] = 1;
			y.push_back(y_matrix);
			// x mit 4 Merkmalen
			x.push_back(Matrix<float>(line.substr(0, line.size()-2)));
		}
		file_train.close();
	}
	else 
		cout << "Error: Die Iris-Datei kann nicht geöffnet werden." << endl;

	// train/test split
	std::size_t const split_pos = int(train_test_split*x.size());
	
	return vector<vector<Matrix<float>>>{
		vector<Matrix<float>>(x.begin(), x.begin() + split_pos),	// x_train
		vector<Matrix<float>>(y.begin(), y.begin() + split_pos),	// y_train
		vector<Matrix<float>>(x.begin() + split_pos, x.end()),		// x_test
		vector<Matrix<float>>(y.begin() + split_pos, y.end())		// y_tets
	};
}