import os
from typing import Dict, List, Optional

import ase
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
)
from metatensor.torch.atomistic.ase_calculator import _compute_ase_neighbors

from featomic.torch import SoapPowerSpectrum, systems_to_torch


HYPERS = {
    "cutoff": {
        "radius": 3.6,
        "smoothing": {"type": "ShiftedCosine", "width": 0.3},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.2,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 3,
        "radial": {"type": "Gto", "max_radial": 11},
    },
}


class Model(torch.nn.Module):
    def __init__(self, types: List[int]):
        super().__init__()
        self.calculator = SoapPowerSpectrum(**HYPERS)
        self.neighbor_types = Labels(
            ["neighbor_1_type", "neighbor_2_type"],
            torch.tensor([(t1, t2) for t1 in types for t2 in types if t1 < t2]),
        )

        n_types = len(types)
        max_radial = HYPERS["basis"]["radial"]["max_radial"]
        max_angular = HYPERS["basis"]["max_angular"]
        in_features = (
            (n_types * (n_types + 1)) * (max_radial + 1) ** 2 // 4 * (max_angular + 1)
        )

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=1, dtype=torch.float64
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "energy" not in outputs:
            return {}

        options = outputs["energy"]

        soap = self.calculator(systems=systems, selected_samples=selected_atoms)
        soap = soap.keys_to_properties(self.neighbor_types)
        soap = soap.keys_to_samples("center_type")

        features = soap.block().values

        if options.per_atom:
            samples = soap.block().samples
        else:
            features = soap.block().values.sum(dim=0, keepdim=True)
            samples = Labels(
                ["system"],
                torch.arange(len(systems), dtype=torch.int32).reshape(-1, 1),
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
    model = Model(types=[1, 6, 8])
    model.eval()

    energy_output = ModelOutput(quantity="energy", unit="eV")
    capabilities = ModelCapabilities(
        supported_devices=["cpu"],
        length_unit="A",
        interaction_range=HYPERS["cutoff"]["radius"],
        atomic_types=[1, 6, 8],
        dtype="float64",
        outputs={"energy": energy_output},
    )
    export = MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    # Check we are requesting the right set of neighbors
    requests = export.requested_neighbor_lists()
    assert len(requests) == 1
    assert requests[0].cutoff == HYPERS["cutoff"]["radius"]
    assert not requests[0].full_list
    assert requests[0].requestors() == ["featomic", "Model.calculator"]

    # check we can save the model
    export.save(os.path.join(tmpdir, "model.pt"))

    # check that we can run the model
    frame = ase.Atoms(
        numbers=[1, 8], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], pbc=False
    )
    system = systems_to_torch(frame)
    neighbors = _compute_ase_neighbors(
        frame, requests[0], dtype=torch.float64, device="cpu"
    )
    system.add_neighbor_list(requests[0], neighbors)

    options = ModelEvaluationOptions(
        length_unit="", outputs={"energy": energy_output}, selected_atoms=None
    )
    _ = export([system], options, check_consistency=True)
