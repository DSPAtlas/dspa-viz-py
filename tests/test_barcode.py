import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
from your_module import BarcodePlotter

class TestBarcodePlotter(unittest.TestCase):

    def setUp(self):
        # Mock data for testing
        self.proteins_in_file_fasta_dict = {
            "protein1": MagicMock(seq="ABCDEFG"),
            "protein2": MagicMock(seq="HIJKLMN")
        }
        self.yeast_proteins = {"protein1", "protein2"}
        self.plotter = BarcodePlotter(
            self.proteins_in_file_fasta_dict, self.yeast_proteins
        )

    def test_sort_dataframe_by_start(self):
        df = pd.DataFrame({
            'start': [3, 1, 2],
            'end': [5, 4, 6],
            'diff': [0.5, -0.8, 1.2]
        })
        sorted_df = self.plotter.sort_dataframe_by_start(df)
        expected_sorted_df = pd.DataFrame({
            'start': [1, 2, 3],
            'end': [4, 6, 5],
            'diff': [-0.8, 1.2, 0.5]
        })
        pd.testing.assert_frame_equal(sorted_df, expected_sorted_df)

    def test_create_empty_matrices(self):
        aa_seq = "ABCDEFG"
        dataframe, len_vector = self.plotter.create_empty_matrices(aa_seq)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertIsInstance(len_vector, np.ndarray)
        self.assertEqual(len(len_vector), len(aa_seq))

    def test_update_len_vector(self):
        len_vector = np.zeros(5)
        self.plotter.update_len_vector(len_vector, 1, 4, 0.7, 0.05, 0.2)
        expected_len_vector = np.array([0, 0.7, 0.7, 0.7, 0])
        np.testing.assert_array_almost_equal(len_vector, expected_len_vector)

    def test_plot_barcode_return_type(self):
        prot = "protein1"
        aa_seq = "ABCDEFG"
        len_vector = np.zeros(len(aa_seq))
        plot = self.plotter.plot_barcode(prot, aa_seq, len_vector)
        self.assertIsInstance(plot, plt.Figure)

    def test_plot_barcode_save_as_svg(self):
        prot = "protein2"
        aa_seq = "HIJKLMN"
        len_vector = np.zeros(len(aa_seq))
        svg_file = self.plotter.plot_barcode(prot, aa_seq, len_vector, save_as_svg=True)
        self.assertTrue(svg_file.endswith(".svg"))

    def test_plot_dynamics_barcode_return_type(self):
        prot = "protein1"
        LiP_df = pd.DataFrame({
            'pg_protein_accessions': ["protein1"] * 3,
            'start': [1, 2, 3],
            'end': [4, 5, 6],
            'diff': [0.5, -0.8, 1.2],
            'adj_pval': [0.01, 0.1, 0.02]
        })
        plot = self.plotter.plot_dynamics_barcode(prot, LiP_df, 0.05, 0.2)
        self.assertIsInstance(plot, plt.Figure)

    def test_plot_residuelevel_barcode_return_type(self):
        prot = "protein2"
        LiP_df = pd.DataFrame({
            'pg_protein_accessions': ["protein2"] * 3,
            'start': [1, 2, 3],
            'end': [4, 5, 6],
            'diff': [0.5, -0.8, 1.2],
            'adj_pval': [0.01, 0.1, 0.02]
        })
        plot = self.plotter.plot_residuelevel_barcode(prot, LiP_df, 0.05, 0.2)
        self.assertIsInstance(plot, plt.Figure)

if __name__ == '__main__':
    unittest.main()