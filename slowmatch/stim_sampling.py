from typing import Optional, Union, Tuple, List

import typer
import stim
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from slowmatch.exposed import Matching
from slowmatch.mwpm import Mwpm


def predict_observable_errors_using_slowmatch(circuit: stim.Circuit,
                                              det_samples: np.ndarray,
                                              verbose: bool = False,
                                              enable_logger: bool = False
                                              ) -> Union[np.ndarray, Tuple[np.ndarray, Matching]]:
    """Turn detection events into predicted observable errors."""
    error_model = circuit.detector_error_model(decompose_errors=True)
    matching_graph = Matching(error_model, enable_logger=enable_logger)

    num_shots = det_samples.shape[0]
    num_obs = circuit.num_observables
    num_dets = circuit.num_detectors
    assert det_samples.shape[1] == num_dets

    predictions = np.zeros(shape=(num_shots, num_obs), dtype=np.bool8)
    myrange = range if not verbose else trange
    for k in myrange(num_shots):
        expanded_det = np.resize(det_samples[k], num_dets + 1)
        expanded_det[-1] = 0
        out = matching_graph.decode(expanded_det)
        predictions[k] = out.predicted_observables
    return predictions, matching_graph


def count_logical_errors(
        circuit: stim.Circuit,
        num_shots: int,
        seed: int = 0,
        verbose: bool = False,
        enable_logger: bool = False
    ) -> Union[int, Tuple[int, List[Mwpm]]]:
    shots = circuit.compile_detector_sampler(seed=seed).sample(num_shots, append_observables=True)

    detector_parts = shots[:, :circuit.num_detectors]
    actual_observable_parts = shots[:, circuit.num_detectors:]
    predicted_observable_parts, matching = predict_observable_errors_using_slowmatch(
                 circuit,
                 detector_parts,
                 verbose=verbose,
                 enable_logger=enable_logger
    )
    num_errors = 0
    for actual, predicted in zip(actual_observable_parts, predicted_observable_parts):
        if not np.array_equal(actual, predicted):
            num_errors += 1
    return num_errors, matching


def repetition_code_threshold():
    seed = 0
    num_shots = 10000
    for d in [3, 5, 7]:
        print(f"\nd={d}", flush=True)
        xs = []
        ys = []
        print(f"Noise: ", end="", flush=True)
        for noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f"{noise} ", end="", flush=True)
            circuit = stim.Circuit.generated(
                "repetition_code:memory",
                rounds=d * 3,
                distance=d,
                before_round_data_depolarization=noise)
            xs.append(noise)
            ys.append(count_logical_errors(circuit, num_shots, seed=seed) / num_shots)
        plt.plot(xs, ys, label="d=" + str(d))
    plt.semilogy()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate")
    plt.legend()
    plt.show()


def surface_code_threshold_example():
    num_shots = 1000
    seed = 0
    ds = [3, 5, 7]
    noise_list = [0.0025, 0.0050, 0.0075, 0.0100]
    for d in ds:
        print(f"\nd={d}", flush=True)
        xs = []
        ys = []
        print(f"Noise: ", end="", flush=True)
        for noise in noise_list:
            print(f"{noise} ", end="", flush=True)
            circuit = stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                rounds=d * 3,
                distance=d,
                after_clifford_depolarization=noise,
                after_reset_flip_probability=noise,
                before_measure_flip_probability=noise,
                before_round_data_depolarization=noise)
            xs.append(noise)
            ys.append(count_logical_errors(circuit, num_shots, seed=seed) / num_shots)
        plt.plot(xs, ys, label="d=" + str(d))
    plt.semilogy()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate")
    plt.legend()
    plt.show()


def run_surface_code_experiment(distance: int, p: float, num_shots: int, seed: Optional[int] = None,
                                verbose: bool = True) -> None:
    circuit = stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        rounds=distance,
        distance=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p)
    num_errs, matching = count_logical_errors(circuit, num_shots, seed=seed, verbose=verbose)
    print(f"{num_errs} {num_shots}")


if __name__ == "__main__":
    typer.run(run_surface_code_experiment)
