# %%
import pandas as pd

# %%
lookup_db = pd.read_csv('data/lookup_db.csv')
selected_alleles = pd.read_csv("data/selected_alleles.csv", header=None)[0].values

# %%
lookup_score_df = pd.merge(lookup_db.loc[lookup_db['allele'].isin(selected_alleles)], peptide_df, how='inner')
index = pd.MultiIndex.from_tuples(
    [[allele, allele[4], s] for allele in selected_alleles for s in peptide_df['sample'].unique()], 
    names=['allele', 'locus', 'sample'])

lookup_score_df = lookup_score_df[['locus', 'sample', 'allele']].value_counts().reset_index()
lookup_score_df['probability'] = np.cbrt(lookup_score_df['count'])
lookup_score_df['probability'] = lookup_score_df.groupby(['sample', 'locus'])['probability'].transform(
    lambda x: x / x.max()
    )
lookup_score_df = lookup_score_df.set_index(['allele', 'locus', 'sample']).reindex(index).fillna(0).reset_index()

def get_homozygous_scores(locus, sample):
    scores = lookup_score_df.loc[
        (lookup_score_df['sample'] == sample) & (lookup_score_df['locus'] == locus), 'count']
    return (scores.nlargest(2).values[-1] <= scores.max() * LOOKUP_HOMOZYGOUS_THRESHOLDS[locus]) * 1

homozygous_scores = pd.DataFrame(
    [[s, (l := p[4]), p, get_homozygous_scores(l, s)] for p in PLACEHOLDERS for s in
        lookup_score_df['sample'].unique()],
    columns=['sample', 'locus', 'allele', 'probability']
)
lookup_score_df = pd.concat([lookup_score_df, homozygous_scores]).drop('count', axis=1)



# %%
