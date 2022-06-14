#pragma once

#include <vector>
#include <array>

const std::string output_y = "./final_y";
const std::string output_var = "./final_var";
const std::string file_residulas = "./residuals";
const std::string file_variances = "./variances";
const std::string extension = ".txt";

const std::size_t DATA_LEN{ 200 };
const std::size_t RANK{ 2 };				// K
const std::array<int, 2> armaRank{ 2, 2 };	// R,S
const std::array<int, 2> garchRank{ 2, 2 }; // Q,P
const std::vector<double> init_ARMA_param{ 0.5,0.5 };		// a,b
const std::vector<double> init_GARCH_param{ 0.5,0.5,0.5 };	// d0, delta, beta
const double conversion_factor{ 100 };
