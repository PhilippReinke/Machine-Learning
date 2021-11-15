#include <algorithm>
#include <random>

#include "Classifier.h"

int main()
{
	// Iris laden
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
			// 
			x.push_back(Matrix<float>(line.substr(0, line.size()-2)));
		}
		file_train.close();
	}
	else 
		cout << "Error: Eine Datei kann nicht geöffnet werden." << endl;

	// train/test split
	std::size_t const split_pos = int(x.size()/2);
	vector<Matrix<float>> x_train(x.begin(), x.begin() + split_pos);
	vector<Matrix<float>> x_test(x.begin() + split_pos, x.end());
	vector<Matrix<float>> y_train(y.begin(), y.begin() + split_pos);
	vector<Matrix<float>> y_test(y.begin() + split_pos, y.end());

	// Lineare Klassifizierung
	auto w_lin = PolynomialClassification(x_train, y_train, 1); w_lin.save("saved_matrices/lin_class.mat");
	// auto w_lin = Matrix<float>("saved_matrices/lin_class.mat");
	// Performancecheck
	auto pred_lin = PolynomialClassification(w_lin, x_train, 1);
	auto res_lin = evaluate_prediction(y_train, pred_lin);
	//cout << res_lin.first << " / " << res_lin.second << " Richtige für Trainingsdaten." << endl;
	//
	auto pred_lin_test = PolynomialClassification(w_lin, x_test, 1);
	auto res_lin_test = evaluate_prediction(y_test, pred_lin_test);
	//cout << res_lin_test.first << " / " << res_lin_test.second << " Richtige für Testdaten.\n" << endl;
	cout << "Lineare Klassifizierung     : Performance " << res_lin_test.first / (float) res_lin_test.second << endl;

	// Polynomiale Klassifizierung
	float best_ratio_poly = -1.0; unsigned best_degree_poly;
	for (unsigned degree=2; degree<=10; degree++)
	{
		auto w_poly = PolynomialClassification(x_train, y_train, degree);
		// w_poly.save("saved_matrices/poly_class.mat");
		// auto w_poly = Matrix<float>("saved_matrices/poly_class.mat");
		// Performancecheck
		auto pred_poly = PolynomialClassification(w_poly, x_train, degree);
		auto res_poly = evaluate_prediction(y_train, pred_poly);
		//cout << res_poly.first << " / " << res_poly.second << " Richtige für Trainingsdaten." << endl;
		//
		auto pred_poly_test = PolynomialClassification(w_poly, x_test, degree);
		auto res_poly_test = evaluate_prediction(y_test, pred_poly_test);
		//cout << res_poly_test.first << " / " << res_poly_test.second << " Richtige für Testdaten.\n" << endl;
		//
		float current_ratio = res_poly_test.first / (float) res_poly_test.second;
		if (current_ratio > best_ratio_poly) { best_ratio_poly = current_ratio; best_degree_poly = degree; }
	}
	cout << "Polynomiale Klassifizierung : Performance " << best_ratio_poly << " mit Grad " << best_degree_poly << endl;

	// Fourier Klassifizierung
	float best_ratio_fourier = -1.0; unsigned best_degree_fourier;
	for (unsigned degree=2; degree<=10; degree++)
	{
		auto w_fourier = FourierClassification(x_train, y_train, degree); w_fourier.save("saved_matrices/fourier_class.mat");
		// auto w_fourier = Matrix<float>("saved_matrices/fourier_class.mat");
		// Performancecheck
		auto pred_fourier = FourierClassification(w_fourier, x_train, degree);
		auto res_fourier = evaluate_prediction(y_train, pred_fourier);
		// cout << res_fourier.first << " / " << res_fourier.second << " Richtige für Trainingsdaten." << endl;
		//
		auto pred_fourier_test = FourierClassification(w_fourier, x_test, degree);
		auto res_fourier_test = evaluate_prediction(y_test, pred_fourier_test);
		// cout << res_fourier_test.first << " / " << res_fourier_test.second << " Richtige für Testdaten.\n" << endl;
		float current_ratio = res_fourier_test.first / (float) res_fourier_test.second;
		if (current_ratio > best_ratio_fourier) { best_ratio_fourier = current_ratio; best_degree_fourier = degree; }
	}
	cout << "Fourier Klassifizierung     : Performance " << best_ratio_fourier << " mit Grad " << best_degree_fourier << endl;

	return 0;
}