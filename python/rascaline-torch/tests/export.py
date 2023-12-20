import os
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelOutput,
    System,
)

from rascaline.torch import SoapPowerSpectrum


HYPERS = {
    "cutoff": 3.6,
    "max_radial": 12,
    "max_angular": 3,
    "atomic_gaussian_width": 0.2,
    "center_atom_weight": 1.0,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.3}},
}


class Model(torch.nn.Module):
    def __init__(self, species: List[int]):
        super().__init__()
        self.calculator = SoapPowerSpectrum(**HYPERS)
        self.species_neighbors = torch.IntTensor(
            [(s1, s2) for s1 in species for s2 in species if s1 < s2]
        )

        n_max = HYPERS["max_radial"]
        l_max = HYPERS["max_angular"]
        in_features = (n_max * len(species)) ** 2 * l_max
        self.linear = torch.nn.Linear(in_features=in_features, out_features=1)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "energy" not in outputs:
            return {}

        options = outputs["energy"]

        if selected_atoms is not None:
            # TODO: change rascaline names to match metatensor
            selected_atoms = selected_atoms.rename("system", "structure")
            selected_atoms = selected_atoms.rename("atom", "center")

        soap = self.calculator(
            systems=systems,
            selected_samples=selected_atoms,
        )

        soap = soap.keys_to_properties(
            Labels(["species_neighbor_1", "species_neighbor_2"], self.species_neighbors)
        )
        soap = soap.keys_to_samples("species_center")

        features = soap.block().values

        if options.per_atom:
            samples = soap.block().samples
            # TODO: change rascaline names to match metatensor
            samples = samples.rename("structure", "system")
            samples = samples.rename("center", "atom")
        else:
            features = soap.block().values.sum(dim=0, keepdim=True)
            samples = Labels(
                ["structure"],
                torch.arange(len(systems), dtype=torch.int32),
            )

        block = TensorBlock(
            values=self.linear(features),
            samples=samples,
            components=[],
            properties=Labels(["energy"], torch.IntTensor([[0]])),
        )

        return {
            "energy": TensorMap(Labels(["_"], torch.IntTensor([[0]])), [block]),
        }


def test_export_as_metatensor_model(tmpdir):
    model = Model(species=[1, 6, 8])
    model.eval()

    export = MetatensorAtomisticModel(model, ModelCapabilities())

    # Check we are requesting the right set of neighbors
    neighbors = export.requested_neighbors_lists()
    assert len(neighbors) == 1
    assert neighbors[0].model_cutoff == HYPERS["cutoff"]
    assert not neighbors[0].full_list
    assert neighbors[0].requestors() == ["rascaline", "Model.calculator"]

    # check we can save the model
    export.export(os.path.join(tmpdir, "model.pt"))
