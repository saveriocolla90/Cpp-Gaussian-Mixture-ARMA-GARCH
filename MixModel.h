#pragma once

#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <math.h>
#include "Utils.h"
#include "Model.h"
#include "ArmaModel.h"
#include "GarchModel.h"

class MixtureModel : public Model {
private:
	int mix_rank_;
	std::array<int, 2> arma_rank_, garch_rank_;
	vec_double alpha_;
	vecvec_double pZ_;
	std::vector<matrix> hessianArma_;
	std::vector<matrix> hessianGarch_;
	std::vector<ArmaModel> Arma_;
	std::vector<GarchModel> Garch_;
public:
	MixtureModel(int _N, int _k, std::array<int, 2> _armaRank, std::array<int, 2> _garchRank) : Model() {
		this->length_ = _N;
		this->mix_rank_ = _k;
		this->arma_rank_ = _armaRank;
		this->garch_rank_ = _garchRank;
		this->pZ_.resize(_k);
		this->setParams(std::array<int, 2>{_k, 0});
		this->setHessianSize();
		this->rndInit();
	};

	/* Output */
	ArmaModel getArma(int rank_) {
		return this->Arma_.at(rank_);
	}
	GarchModel getGarch(int rank_) {
		return this->Garch_.at(rank_);
	}
	void output(int _none = 0) override {

		// Print each model series
		for (int k = 0; k < this->mix_rank_; k++) {
			this->Arma_.at(k).output(k);
			this->Garch_.at(k).output(k);
		}

		// Compute the final fit of data and print it 
		TS y_expected;
		TS var_expected = this->computeFitData(y_expected);

		// Print final result to file
		WriteTStoFile(y_expected, output_y + extension);
		WriteTStoFile(var_expected, output_var + extension);
	};

	void setData(TS _ts_in) override {
		/* Set data for all models */

		for (auto& mod : Arma_)
			mod.setData(_ts_in);
	};
	void setParams(std::array<int, 2> _rank) override {
		for (int k = 0; k < _rank.at(0); k++) {
			this->alpha_.push_back(1. / double(mix_rank_));
			this->pZ_.at(k).resize(this->length_ - arma_rank_.at(0) - garch_rank_.at(0));
		}
	};
	void rndInit() override {
		// Initilize random initial ARMA-GARCH models
		int R = arma_rank_.at(0);
		int Q = garch_rank_.at(0);
		for (int k = 0; k < this->mix_rank_; k++) {
			ArmaModel new_arma(arma_rank_, this->length_ - R);
			GarchModel new_garch(garch_rank_, this->length_ - R - Q);
			Arma_.push_back(new_arma);
			Garch_.push_back(new_garch);
		}
	};
	bool checkStability() override {
		// Each model has to be a stable one
		for (auto& mod : this->Garch_)
			while (not mod.checkStability())
				mod.rndInit();

		return true;
	};
	void setHessianSize() {
		this->hessianArma_.resize(mix_rank_);
		this->hessianGarch_.resize(mix_rank_);

		// For each mix model component K expand a 2x2 matrix for each parameter
		for (int k = 0; k < mix_rank_; k++) {
			for (int i = 0; i < 2; i++) {
				this->hessianArma_.at(k).at(i).resize(arma_rank_.at(i));
				this->hessianGarch_.at(k).at(i).resize(garch_rank_.at(i));
			}
			for (int i = 0; i < 2; i++) {
				for (auto& vec : this->hessianArma_.at(k).at(i))
					vec.resize(arma_rank_.at(i));
				for (auto& vec : this->hessianGarch_.at(k).at(i))
					vec.resize(garch_rank_.at(i));
			}
		}
	}

	/* Compute EM steps */
	void compute() override {
		/* Compute residuals for all models */
		int k = 0;
		for (auto& mod : Arma_) {
			mod.compute();

			// Set residuals as data series for Garch model
			Garch_.at(k++).setData(mod.getSeries());
		}

		/* Compute variances for all models */
		for (auto& mod : Garch_)
			mod.compute();
	};

	/* Derivatives */
	void derivate() {
		/* ARMA derivatives for A and B parameters */
		for (auto& mod : Arma_)
			mod.derivate();

		int k = 0;
		for (auto& mod : Garch_) {
			auto ts_in_a = Arma_.at(k).getDerivate_a();
			auto ts_in_b = Arma_.at(k).getDerivate_b();
			auto resid = Arma_.at(k).getResiduals();
			mod.derivate(ts_in_a, ts_in_b, resid);
			k++;
		}
	}
	void mixDerivateArma() {

		// Compute 2nd order partial derivatives for each Gaussian Mixture component
		for (int k = 0; k < this->mix_rank_; k++) {
			// Set the K component's hessian matrix for ARMA model
			auto& hessian = this->hessianArma_.at(k);

			// Retrieve data: partial derivatives respect A and variance series
			auto d_resid = Arma_.at(k).getDerivate_a();
			auto d_var = Garch_.at(k).getDerivate_a();
			auto var = Garch_.at(k).getVariance();
			const double N = var.size();
			const int offset = d_resid.at(0).size() - int(N);

			// Respect parameter A
			for (int i = 0; i < hessian.at(ArmaDerivate::A).size(); i++) {
				for (int j = 0; j < hessian.at(ArmaDerivate::A).at(i).size(); j++) {
					auto part_resid = ((d_resid.at(i) * d_resid.at(j))>>offset) / var;		// Partial residuals A
					auto part_var = d_var.at(i) * d_var.at(j) * (1. / (2. * (var ^ 2.)));	// Partial variance A
					auto total = (part_resid + part_var) * this->pZ_.at(k);					// Multiply by likelihood pZ

					// Rescale to variance lenght
					hessian.at(ArmaDerivate::A).at(i).at(j) = std::accumulate(total.begin(), total.end(), 0.) / N;
				}
			}

			// Retrieve data: partial derivatives respect B
			d_resid = Arma_.at(k).getDerivate_b();
			d_var = Garch_.at(k).getDerivate_b();

			// Respect parameter B
			for (int i = 0; i < hessian.at(ArmaDerivate::B).size(); i++) {
				for (int j = 0; j < hessian.at(ArmaDerivate::B).at(i).size(); j++) {
					auto part_resid = ((d_resid.at(i) * d_resid.at(j))>>offset) / var;		// Partial residuals B
					auto part_var = d_var.at(i) * d_var.at(j) * (1. / (2. * (var ^ 2.)));	// Partial variance B
					auto total = (part_resid + part_var) * this->pZ_.at(k);					// Multiply by likelihood pZ

					// Rescale to variance lenght
					hessian.at(ArmaDerivate::B).at(i).at(j) = std::accumulate(total.begin(), total.end(), 0.) / N;
				}
			}
		}
	}
	void mixDerivateGarch() {
		// Compute 2nd order partial derivatives for each Gaussian Mixture component
		for (int k = 0; k < this->mix_rank_; k++) {
			// Set the K component's hessian matrix for GARCH model
			auto& hessian = this->hessianGarch_.at(k);

			// Retrieve data: partial derivatives respect GAMMA and variance series
			auto d_var = Garch_.at(k).getDerivate_gamma();
			auto var = Garch_.at(k).getVariance();
			auto pZ = this->pZ_.at(k);
			const int offset = Garch_.at(k).getRank()[0];
			const double N = var.size();

			// Respect parameter GAMMA
			for (int i = 0; i < hessian.at(GarchDerivate::Gamma).size(); i++) {
				for (int j = 0; j < hessian.at(GarchDerivate::Gamma).at(i).size(); j++) {
					auto part_var = d_var.at(i) * d_var.at(j) * (1. / (2. * (var ^ 2.)));	// Partial variance GAMMA
					auto total = part_var * pZ;									// Multiply by likelihood pZ

					// Rescale to variance lenght
					hessian.at(GarchDerivate::Gamma).at(i).at(j) = std::accumulate(total.begin(), total.end(), 0.) / N;
				}
			}

			// Retrieve data: partial derivatives respect RHO and variance series
			d_var = Garch_.at(k).getDerivate_rho();
			var = var >> offset;
			pZ = pZ >> offset;

			// Respect parameter RHO
			for (int i = 0; i < hessian.at(GarchDerivate::Rho).size(); i++) {
				for (int j = 0; j < hessian.at(GarchDerivate::Rho).at(i).size(); j++) {
					auto part_var = d_var.at(i) * d_var.at(j) * (1. / (2. * (var ^ 2.)));	// Partial variance RHO
					auto total = part_var * pZ;									// Multiply by likelihood pZ

					// Rescale to variance lenght
					hessian.at(GarchDerivate::Rho).at(i).at(j) = std::accumulate(total.begin(), total.end(), 0.) / N;
				}
			}
		}
	}

	/* Probabilities & Likelihoods */
	void compute_pZ() {
		// Normalization
		vec_double norm(this->Garch_.at(0).getVariance().size());
		int offset = this->Arma_.at(0).getRank()[1];
		for (int k = 0; k < mix_rank_; k++) {
			const auto& resid = this->Arma_.at(k).getResiduals();
			const auto& var = this->Garch_.at(k).getVariance();
			auto partial_sum = this->alpha_.at(k) * normal_pdf((resid>>offset) / (var ^ 0.5)) / (var ^ 0.5);
			norm = norm + partial_sum;
		}

		// Compute unobserved Z components likelihoods
		for (int k = 0; k < mix_rank_; k++) {
			const auto& resid = this->Arma_.at(k).getResiduals();
			const auto& var = this->Garch_.at(k).getVariance();
			this->pZ_.at(k) = (this->alpha_.at(k) / (var ^ 0.5)) * normal_pdf((resid>>offset) / (var ^ 0.5)) / norm;
		}
	}
	double compute_Qfunc() {
		/* Compute the expectation value Q */

		vec_double Q_star(this->length_ - this->arma_rank_.at(0) - this->garch_rank_.at(0));
		for (int k = 0; k < mix_rank_; k++) {
			auto resid = this->Arma_.at(k).getResiduals();
			auto var = this->Garch_.at(k).getVariance();
			int offset = resid.size() - var.size();
			Q_star = Q_star + this->pZ_.at(k) * log(this->alpha_.at(k) * normal_pdf((resid >> offset) / (var ^ 0.5)) / (var ^ 0.5));
		}

		return std::accumulate(Q_star.begin(), Q_star.end(), 0.);
	}
	void compute_Alpha() {
		/* Compute new alpha weights by pZ values*/

		double N = this->length_ - this->arma_rank_.at(0) - this->garch_rank_.at(0);
		for (int k = 0; k < mix_rank_; k++) {
			auto& pZ = this->pZ_.at(k);
			this->alpha_.at(k) = std::accumulate(pZ.begin(), pZ.end(), 0.) / N;
		}
	}
	vec_double ll_gradient(vecvec_double& _d_resid, vecvec_double& _d_var, int& _k_comp) {
		/* Compute the log-likelihood gradient using the input derivatives
		   passed according to the reference parameter which they are computed by. */

		vec_double ll_grad;
		int p_rank = _d_resid.size();
		ll_grad.resize(p_rank);

		// Get data from K-th model
		auto resid = this->Arma_.at(_k_comp).getResiduals();
		auto var = this->Garch_.at(_k_comp).getVariance();
		auto pZ = this->pZ_.at(_k_comp);
		double N = var.size();
		int offset = resid.size() - int(N);
		resid = resid >> offset;

		// Iterate over all the parameters 
		for (int p = 0; p < p_rank; p++) {
			// Residuals and Variance partials
			auto part_resid = pZ * resid * _d_resid.at(p) / var;
			auto part_var = ((0.5 * pZ) / var) * (( (resid ^ 2.) / var ) - 1.) * _d_var.at(p);

			// Get total sum
			auto total_resid = std::accumulate(part_resid.begin(), part_resid.end(), 0.);
			auto total_var = std::accumulate(part_var.begin(), part_var.end(), 0.);

			// Compute log-likelihood gradient 
			ll_grad.at(p) = (total_var - total_resid) / N;
		}

		return ll_grad;
	}
	vec_double ll_gradient(vecvec_double& _d_var, int& _k_comp) {
		/* Compute the log-likelihood gradient using the input derivatives
		   passed according to the reference parameter which they are computed by:
			- for a fixed parameter we have k components models
			- when the residuals part is missing we compute only the variance gradient
		*/

		vec_double ll_grad;
		int p_rank = _d_var.size();
		ll_grad.resize(p_rank);

		// Get data from K-th model
		auto resid = this->Arma_.at(_k_comp).getResiduals();
		auto var = this->Garch_.at(_k_comp).getVariance();
		auto pZ = this->pZ_.at(_k_comp);
		double N = _d_var.at(0).size();
		int offset = resid.size() - int(N);
		resid = resid >> offset;
		if (var.size() > N)
			var = var >> int(var.size() - N);
		if (pZ.size() > N)
			pZ = pZ >> int(pZ.size() - N);


		// Iterate over all the parameters 
		for (int p = 0; p < p_rank; p++) {	
			// Variance partials
			auto part_var = ((0.5 * pZ) / var) * (( (resid ^ 2.) / var) - 1.) * _d_var.at(p);

			// Get total sum
			auto total_var = std::accumulate(part_var.begin(), part_var.end(), 0.);

			// Compute log-likelihood gradient 
			ll_grad.at(p) = total_var / N;
		}

		return ll_grad;
	}
	double ll_gradient(vec_double& _d_var, int& _k_comp) {
		/* Compute the log-likelihood gradient using the input derivatives
		   passed according to the reference parameter which they are computed by. */

		double ll_grad;

		// Get data from K-th model
		auto resid = this->Arma_.at(_k_comp).getResiduals();
		auto var = this->Garch_.at(_k_comp).getVariance();
		auto& pZ = this->pZ_.at(_k_comp);
		double N = var.size();
		int offset = resid.size() - int(N);
		resid = resid >> offset;

		// Variance partials
		auto part_var = ((0.5 * pZ) / var) * (( (resid ^ 2.) / var) - 1.) * _d_var;

		// Get total sum
		auto total_var = std::accumulate(part_var.begin(), part_var.end(), 0.);

		// Compute log-likelihood gradient 
		ll_grad = total_var / N;
		
		return ll_grad;
	}

	/* Fitting procedure */
	void fittingProc() {
		/* Here define a subroutine to put all the derivatives togetherand compute the parameters increment
			- First compute the partial derivatives for residuals(dr) and variancies(dv)
			- Then compute the hessian matrix(H)
			- Finally compute the matrix products : H ^ -1 * dr, H ^ -1 * dv 

		....(for ARMA and GARCH parameters respectively)
		*/

		// Increment ARMA parameters
		this->paramIncr_ARMA();

		// Update the current model series (y_hat, residuals and variances)
		int k = 0;
		for (auto& mod : Arma_) {
			mod.compute();

			// Set residuals as data series for Garch model
			Garch_.at(k++).setData(mod.getSeries());
		}

		/* Compute variances for all models */
		for (auto& mod : Garch_)
			mod.compute();

		// Store current parsmeter values for future restoring
		vecvec_double delta_reset(this->mix_rank_);
		vecvec_double beta_reset(this->mix_rank_);
		for (int k = 0; k < this->mix_rank_; k++) {
			delta_reset.at(k) = this->Garch_.at(k).getParams(GarchParams::Delta);
			beta_reset.at(k) = this->Garch_.at(k).getParams(GarchParams::Beta);
		}

		// Increment GARCH parameters
		this->paramIncr_GARCH();

		// Update the current model series (variances)
		for (int k = 0; k < mix_rank_; k++) {
			if (this->Garch_.at(k).checkStability()) {
				this->Garch_.at(k).compute();
			}
			else {
				this->Garch_.at(k).resetParmas(delta_reset.at(k), beta_reset.at(k));
			}
		}
	}
	void paramIncr_ARMA() {
		/* Increment the ARMA parameters a & b adopting the Newton's optimization method. */

		// Compute the Hessian matrix 2nd order log-likelihood derivatives
		this->mixDerivateArma();

		// Compute the loglikelihood gradient respect parameter A
		int offset = this->Arma_.at(0).getRank()[0];
		for (int k = 0; k < mix_rank_; k++) {
			auto d_resid_a = this->Arma_.at(k).getDerivate_a();
			d_resid_a = d_resid_a >> offset;
			auto d_var_a = this->Garch_.at(k).getDerivate_a();
			auto d_loglike = this->ll_gradient(d_resid_a, d_var_a, k);

			// Increment parameter B for every k-th component
			try {
				vecvec_double H_inv = matrInverse(this->hessianArma_.at(k).at(ArmaDerivate::A));
				auto incr = matrProd(H_inv, d_loglike);
				this->Arma_.at(k).incrParam(incr, ArmaDerivate::A);
			}
			catch (const std::runtime_error& error) {
				Display("\nParameter A not incremented!");
			}
		}

		// Compute the loglikelihood gradient respect parameter B
		for (int k = 0; k < mix_rank_; k++) {
			auto d_resid_b = this->Arma_.at(k).getDerivate_b();
			d_resid_b = d_resid_b >> offset;
			auto d_var_b = this->Garch_.at(k).getDerivate_b();
			auto d_loglike = this->ll_gradient(d_resid_b, d_var_b, k);

			// Increment parameter B for every k-th component
			try {
				vecvec_double H_inv = matrInverse(this->hessianArma_.at(k).at(ArmaDerivate::B));
				auto incr = matrProd(H_inv, d_loglike);
				this->Arma_.at(k).incrParam(incr, ArmaDerivate::B);

			}
			catch (const std::runtime_error& error) {
				Display("\nParameter B not incremented!");
			}
		}
	}
	void paramIncr_GARCH(){
		/* Increment the GARCH parameters d0, delta & beta adopting the Newton's optimization method. */

		// Compute the Hessian matrix 2nd order log-likelihood derivatives
		this->mixDerivateGarch();

		// First compute gradient for GAMMA0 parameter
		for (int k = 0; k < mix_rank_; k++) {
			auto var = this->Garch_.at(k).getVariance();
			auto d_var_delta0 = this->Garch_.at(k).getDerivate_gamma0();
			auto part_var = (d_var_delta0 ^ 2.) / (2. * (var^2.));
			auto total = part_var * this->pZ_.at(k);

			// The hessian matrix for Gamma0 has only 1 component
			double N = var.size();
			auto hessian = std::accumulate(total.begin(), total.end(), 0.) / N;

			// Compute increment
			auto d_loglike = this->ll_gradient(d_var_delta0, k);
			auto incr = d_loglike / hessian;
			this->Garch_.at(k).incrParam(incr, GarchDerivate::Gamma_0);
		}

		// Compute the loglikelihood gradient respect parameter GAMMA
		for (int k = 0; k < mix_rank_; k++) {
			auto d_var_delta = this->Garch_.at(k).getDerivate_gamma();
			auto d_loglike = this->ll_gradient(d_var_delta, k);

			// Increment parameter GAMMA for every k-th component
			try {
				vecvec_double H_inv = matrInverse(this->hessianGarch_.at(k).at(GarchDerivate::Gamma));
				auto incr = matrProd(H_inv, d_loglike);
				this->Garch_.at(k).incrParam(incr, GarchDerivate::Gamma);
			}
			catch (const std::runtime_error& error) {
				Display("\nParameter GAMMA not incremented!");
			}
		}

		// Compute the loglikelihood gradient respect parameter RHO
		for (int k = 0; k < mix_rank_; k++) {
			auto d_var_beta = this->Garch_.at(k).getDerivate_rho();
			auto d_loglike = this->ll_gradient(d_var_beta, k);

			// Increment parameter B for every k-th component
			try {
				vecvec_double H_inv = matrInverse(this->hessianGarch_.at(k).at(GarchDerivate::Rho));
				auto incr = matrProd(H_inv, d_loglike);
				this->Garch_.at(k).incrParam(incr, GarchDerivate::Rho);
			}
			catch (const std::runtime_error& error) {
				Display("\nParameter RHO not incremented!");
			}
		}

		bool back_to = true;
		for (int k = 0; k < mix_rank_; k++)
			this->Garch_.at(k).transform(back_to);
	}
	TS computeFitData(TS& y_final) {
		/* Compute the final expected values and put them toghter for final result. */

		for (int k = 0; k < this->mix_rank_; k++) {
			this->Arma_.at(k).computeExpected();
			this->Garch_.at(k).computeExpected();
		}

		y_final = this->Arma_.at(0).getModelSeries();
		TS var_1 = this->Garch_.at(0).getModelSeries();
		
		y_final *= this->alpha_.at(0);
		var_1 *= this->alpha_.at(0);
		for (int k = 1; k < this->mix_rank_; k++) {
			y_final += this->alpha_.at(k) * this->Arma_.at(k).getModelSeries();
			var_1 += this->alpha_.at(k) * this->Garch_.at(k).getModelSeries();
		}

		// Add variance second order
		TS var_2 = this->alpha_.at(0) * ((this->Arma_.at(0).getModelSeries() - y_final)^2.);
		for (int k = 1; k < this->mix_rank_; k++) {
			var_2 += this->alpha_.at(k) * ((this->Arma_.at(k).getModelSeries() - y_final)^2.);
		}
		TS var_final = (var_1 + var_2) ^ 0.5;

		return var_final;
	}

	/* Display functions */
	void printParams() {
		/* Print all the current parameters for each model ARMA and GARCH */
		Display("\n=============================");
		Display("ARMA models:");
		Display("=============================\n");
		int k = 0;
		for (auto& mod : this->Arma_) {
			Display("---------------------------");
			Display("Model " + std::to_string(++k) + ")");
			Display("---------------------------");
			mod.printParams();
		}

		k = 0;
		Display("\n=============================");
		Display("GARCH models:");
		Display("=============================\n");
		for (auto& mod : this->Garch_) {
			Display("---------------------------");
			Display("Model " + std::to_string(++k) + ")");
			Display("---------------------------");
			mod.printParams();
		}
	}
};
