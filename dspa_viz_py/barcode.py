from typing import Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

class BarcodePlotter:
    def __init__(self, proteins_in_file_fasta_dict: dict, organism_proteins: set):
        """
        Initialize the BarcodePlotter.

        Parameters:
        - proteins_in_file_fasta_dict (dict): Dictionary mapping protein names to their sequences.
        - organism_proteins (set): Set of organism protein names.
        """
        self.proteins_in_file_fasta_dict = proteins_in_file_fasta_dict
        self.organism_proteins = organism_proteins

    def sort_dataframe_by_start(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the DataFrame by the 'start' column.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: DataFrame sorted by 'start' column.
        """
        return df.sort_values('start')

    def create_empty_matrices(self, aa_seq: str) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Create empty matrices for plotting.

        Parameters:
        - aa_seq (str): Amino acid sequence.

        Returns:
        - tuple[pd.DataFrame, np.ndarray]: Tuple containing DataFrame and NumPy array.
        """
        len_vector = np.zeros(len(aa_seq))
        return pd.DataFrame(index=range(len(aa_seq))), len_vector

    def update_len_vector(
        self,
        start: int,
        end: int,
        log2FC: float,
        qvalue: float,
    ) -> None:
        """
        Update the length vector based on specified criteria.

        Parameters:
        - len_vector (np.ndarray): Length vector to be updated.
        - start (int): Start position.
        - end (int): End position.
        - log2FC (float): Log2 fold change value.
        - qvalue_cutoff (float): Q-value cutoff.
        - log2FC_cutoff (float): Log2 fold change cutoff.
        """
        log2FC = 0 if np.isinf(log2FC) else log2FC
        for i in range(start, end):
            if self.type == "dynamic":
                if qvalue < self.qvalue_cutoff and abs(log2FC)> self.log2FC_cutoff:
                    self.len_vector[i] = np.mean([self.len_vector[i], log2FC]) if self.len_vector[i] != 0 else log2FC
                else:
                    self.len_vector_detected[i] = 1
            else:
                score = -np.log10(qvalue)+np.abs(log2FC)
                self.len_vector[i] = np.mean([self.len_vector[i], score]) if self.len_vector[i] != 0 else score
    
    def prepare_data(
            self,
            prot:str,
            LiP_df:pd.DataFrame
        ):
        self.aa_seq = self.proteins_in_file_fasta_dict[prot].seq
        
        # Sort the dataframe by start
        LiP_df[LiP_df.pg_protein_accessions == prot].sort_values('start', inplace=True)

        # Create matrices for plot
        self.dataframe_with_vector = pd.DataFrame(index=range(len(self.aa_seq)))
        self.dataframe_with_vector_detected = pd.DataFrame(index=range(len(self.aa_seq)))

        self.len_vector = np.zeros(len(self.aa_seq))
        self.len_vector_detected = np.zeros(len(self.aa_seq))

        for _, row in LiP_df[LiP_df.pg_protein_accessions == prot].iterrows():
                self.update_len_vector( 
                    start=int(row['start']), 
                    end=int(row['end']), 
                    log2FC=row['diff'], 
                    qvalue=row['adj_pval']
                )
        self.dataframe_with_vector[0] = np.where(self.len_vector != 0, self.len_vector, np.nan)
        self.dataframe_with_vector_detected[0] = np.where(
            (self.len_vector_detected == 1) & (self.len_vector == 0), 
            0, 
            np.nan
        )     

    def plot_dynamics_barcode(
        self,
        prot: str,
        LiP_df: pd.DataFrame,
        qvalue_cutoff: float,
        log2FC_cutoff: float,
        save_as_svg: bool = False
    ) -> Union[str, plt.Figure]:
        """
        Plot barcode for structural dynamics of a protein.
        Significantly changing regions (based on defined cut-offs) are coloured in orange,
        identified (but not changing) regions in grey and identified regions are black.

        Parameters:
        - prot (str): Protein name.
        - LiP_df (pd.DataFrame): DataFrame containing structural dynamics data.
        - qvalue_cutoff (float): Q-value cutoff.
        - log2FC_cutoff (float): Log2 fold change cutoff.
        - save_as_svg (bool, optional): Whether to save the plot as SVG. Defaults to False.

        Returns:
        - Union[str, plt.Figure]: Either the name of the saved SVG image or the plot itself.
        """
        self.type = "dynamic"

        if prot not in self.organism_proteins:
            print(f"{prot} not found")
            return
        
        self.qvalue_cutoff = qvalue_cutoff
        self.log2FC_cutoff = log2FC_cutoff

        self.prepare_data(prot=prot, LiP_df=LiP_df)
            
        dataframe_with_vector_barcode = pd.concat(
                [self.dataframe_with_vector, self.dataframe_with_vector_detected], 
                axis=1
        )
        dataframe_with_vector_barcode.columns = ['sig', 'detected']

        # set color maps
        cmap_detected = mpl.colors.LinearSegmentedColormap.from_list("", ["silver", "silver"])
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "black"])
        cmap_sig = mpl.colors.LinearSegmentedColormap.from_list("", ["orange", "orange"])
            
        # Plotting
        fig, ax = plt.subplots(figsize=(6, 1))

        ax.imshow(dataframe_with_vector_barcode[['sig', 'detected']].notna().T,
                    cmap=cmap, aspect='auto', interpolation='nearest')
        ax.imshow(dataframe_with_vector_barcode[['sig']].values.reshape((1, -1)),
                    cmap=cmap_sig, aspect='auto', interpolation='nearest', vmin=0)
        ax.imshow(dataframe_with_vector_barcode[['detected']].values.reshape((1, -1)),
                    cmap=cmap_detected, aspect='auto', interpolation='nearest')
            
        # Set x-axis ticks and labels
        positions = np.arange(0, len(self.aa_seq), 100)
        positions = np.append(positions, [1, len(self.aa_seq)])
        positions = positions[positions != 0]

        ax.set_xticks(positions)
        ax.set_xticklabels(positions)
        ax.set_yticklabels([])
        ax.set_yticks([])

        ax.set_title(prot, fontdict={'fontsize': 16, 'fontweight': 'medium', 'fontname': 'arial'})
        sns.despine(left=True, right=True, top=True)

        if save_as_svg:
            plt.savefig(f'{prot}_structural_dynamics_barcode.svg', format='svg')
            plt.close(fig)
            return f'{prot}_structural_dynamics_barcode.svg'
        else:
            return fig

    def plot_residuelevel_barcode(
        self,
        prot: str,
        LiP_df: pd.DataFrame,
        qvalue_cutoff: float,
        log2FC_cutoff: float,
        save_as_svg: bool = False
    ) -> Union[str, plt.Figure]:
        """
        Plot barcode for residue-level dynamics of a protein. For each residue, the following score is computed:
        mean(-np.log10(qvalue)+np.abs(log2FC)). Residues which are not detected are coloured in grey.

        Parameters:
        - prot (str): Protein name.
        - LiP_df (pd.DataFrame): DataFrame containing residue-level dynamics data.
        - qvalue_cutoff (float): Q-value cutoff.
        - log2FC_cutoff (float): Log2 fold change cutoff.
        - save_as_svg (bool, optional): Whether to save the plot as SVG. Defaults to False.

        Returns:
        - Union[str, plt.Figure]: Either the name of the saved SVG image or the plot itself.
        """
        self.type = "residuelevel"
        if prot not in self.organism_proteins:
            print(f"{prot} not found")
            return
        
        self.prepare_data(prot=prot, LiP_df=LiP_df) 
        # set color maps
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["silver", "silver"])
        # Plotting
        fig, ax = plt.subplots(figsize=(6, 1))

        ax.imshow(
            self.dataframe_with_vector.notna().T,
            cmap=cmap, aspect='auto', interpolation='nearest')
        
        ax.imshow(self.dataframe_with_vector.values.reshape((1, -1)),
            cmap="YlOrBr", aspect='auto', interpolation='nearest', vmin=0)

        # Set x-axis ticks and labels
        positions = np.arange(0, len(self.aa_seq), 100)
        positions = np.append(positions, [1, len(self.aa_seq)])
        positions = positions[positions != 0]

        ax.set_xticks(positions)
        ax.set_xticklabels(positions)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_title(prot, fontdict={'fontsize': 16, 'fontweight': 'medium', 'fontname': 'arial'})
        sns.despine(left=True, right=True, top=True)
        if save_as_svg:
            plt.savefig(f'{prot}_residuelevel_barcode.svg', format='svg')
            plt.close(fig)
            return f'{prot}_residuelevel_barcode.svg'
        else:
            return fig