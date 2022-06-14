
#include <vector>
#include <array>

#include "InputOutput.h"
#include "Model.h"
#include "ArmaModel.h"
#include "GarchModel.h"
#include "MixModel.h"
#include "Settings.h"

int main() {

	Loader L;
	L.setPath("BTTUSDT.txt");
	L.load();
	//L.printData();

	MixtureModel MixModel(DATA_LEN, RANK, armaRank, garchRank);	// K rank mixture model
	//Model* mptr;
	//mptr = &MixModel;
	//mptr->setData(L.getData());
	MixModel.setData(L.getData());

	if (MixModel.checkStability()) {
		Display("Stable initial mixture model!");
		MixModel.printParams();
	}

	/* ----------------This part needs to be iterated untill the convergence of Q value-------------------- */
	for (int i = 0; i < 20; i++) {
		MixModel.compute();
		MixModel.derivate();
		MixModel.compute_pZ();
		MixModel.fittingProc();
		auto Q_value = MixModel.compute_Qfunc();
		Display("\nQ value: " + std::to_string(Q_value));
		MixModel.compute_Alpha();
	}
	/* ------------------------------------------------------------------------------------------------------ */

	MixModel.output();
	Display("\nDone!");
	Display("\n---------------------------------");
	Display("\nParameter's RESULTs:");
	MixModel.printParams();

		

	return 0;
}