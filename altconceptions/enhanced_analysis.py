# enhanced_analysis.py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def statistical_tests(pre_scores, post_scores):
    """
    Perform comprehensive statistical tests on pre-post data
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(post_scores, pre_scores)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(post_scores, pre_scores)
    except:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
    
    # Calculate effect sizes
    n = len(pre_scores)
    cohens_d = (np.mean(post_scores) - np.mean(pre_scores)) / np.std(pre_scores - post_scores, ddof=1)
    
    # Hedges' g (corrected for small sample bias)
    hedges_g = cohens_d * (1 - (3/(4*(n-1) - 1)))
    
    # Percentage of students who improved
    improved = np.sum(post_scores > pre_scores)
    same = np.sum(post_scores == pre_scores)
    declined = np.sum(post_scores < pre_scores)
    
    return {
        'n_students': int(n),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'wilcoxon_statistic': float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else np.nan,
        'wilcoxon_p': float(wilcoxon_p) if not np.isnan(wilcoxon_p) else np.nan,
        'cohens_d': float(cohens_d),
        'hedges_g': float(hedges_g),
        'improved_count': int(improved),
        'improved_pct': float(improved/n*100),
        'same_count': int(same),
        'same_pct': float(same/n*100),
        'declined_count': int(declined),
        'declined_pct': float(declined/n*100)
    }

def correlation_analysis(pre_scores, post_scores):
    """
    Analyze correlations between pre-scores, post-scores, and gains
    """
    gains = post_scores - pre_scores
    
    # Pearson correlations
    pre_post_corr, pre_post_p = pearsonr(pre_scores, post_scores)
    pre_gain_corr, pre_gain_p = pearsonr(pre_scores, gains)
    post_gain_corr, post_gain_p = pearsonr(post_scores, gains)
    
    # Spearman correlations (non-parametric)
    pre_post_spearman, pre_post_spearman_p = spearmanr(pre_scores, post_scores)
    pre_gain_spearman, pre_gain_spearman_p = spearmanr(pre_scores, gains)
    
    return {
        'pearson_pre_post': {'correlation': float(pre_post_corr), 'p_value': float(pre_post_p)},
        'pearson_pre_gain': {'correlation': float(pre_gain_corr), 'p_value': float(pre_gain_p)},
        'pearson_post_gain': {'correlation': float(post_gain_corr), 'p_value': float(post_gain_p)},
        'spearman_pre_post': {'correlation': float(pre_post_spearman), 'p_value': float(pre_post_spearman_p)},
        'spearman_pre_gain': {'correlation': float(pre_gain_spearman), 'p_value': float(pre_gain_spearman_p)}
    }

def concept_level_analysis(merged_df, concept_pairs):
    """
    Enhanced concept-level analysis with effect sizes
    """
    results = []
    
    for label, pre_cols, post_cols in concept_pairs:
        # Handle single or multiple columns
        if isinstance(pre_cols, str):
            pre_data = merged_df[pre_cols]
        else:
            pre_data = merged_df[pre_cols].sum(axis=1)
            
        if isinstance(post_cols, str):
            post_data = merged_df[post_cols]
        else:
            post_data = merged_df[post_cols].sum(axis=1)
        
        # Calculate statistics
        pre_mean = pre_data.mean()
        post_mean = post_data.mean()
        gain = post_mean - pre_mean
        
        # Paired t-test for this concept
        t_stat, p_value = stats.ttest_rel(post_data, pre_data)
        
        # Effect size
        n_concept = len(pre_data.dropna())
        if n_concept > 1:
            sd_pooled = np.sqrt((pre_data.var() + post_data.var()) / 2)
            if sd_pooled > 0:
                cohens_d_concept = gain / sd_pooled
            else:
                cohens_d_concept = np.nan
        else:
            cohens_d_concept = np.nan
        
        # Percentage correct/improved
        if isinstance(pre_cols, str) and pre_data.dtype in [int, float]:
            # For binary items
            pre_correct = pre_data.sum()
            post_correct = post_data.sum()
            retained = ((pre_data == 1) & (post_data == 1)).sum()
            corrected = ((pre_data == 0) & (post_data == 1)).sum()
            regressed = ((pre_data == 1) & (post_data == 0)).sum()
            
            results.append({
                'concept': label,
                'pre_cols': pre_cols if isinstance(pre_cols, str) else '+'.join(pre_cols),
                'post_cols': post_cols if isinstance(post_cols, str) else '+'.join(post_cols),
                'pre_mean': float(pre_mean),
                'post_mean': float(post_mean),
                'mean_gain': float(gain),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d_concept),
                'pre_correct_count': int(pre_correct),
                'pre_correct_pct': float(pre_correct/n_concept*100),
                'post_correct_count': int(post_correct),
                'post_correct_pct': float(post_correct/n_concept*100),
                'retained_count': int(retained),
                'retained_pct': float(retained/pre_correct*100) if pre_correct > 0 else np.nan,
                'corrected_count': int(corrected),
                'corrected_pct': float(corrected/(n_concept-pre_correct)*100) if (n_concept-pre_correct) > 0 else np.nan,
                'regressed_count': int(regressed),
                'regressed_pct': float(regressed/pre_correct*100) if pre_correct > 0 else np.nan
            })
        else:
            # For non-binary or composite scores
            results.append({
                'concept': label,
                'pre_cols': pre_cols if isinstance(pre_cols, str) else '+'.join(pre_cols),
                'post_cols': post_cols if isinstance(post_cols, str) else '+'.join(post_cols),
                'pre_mean': float(pre_mean),
                'post_mean': float(post_mean),
                'mean_gain': float(gain),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d_concept)
            })
    
    return pd.DataFrame(results)

def compute_normalized_gain(pre_scores, post_scores):
    """
    Compute normalized gain and categorize it
    """
    gain = post_scores - pre_scores
    normalized_gain = gain / (10 - pre_scores)  # Assuming max score is 10
    
    # Categorize gains following Hake (1998)
    high_gain = normalized_gain > 0.7
    medium_gain = (normalized_gain >= 0.3) & (normalized_gain <= 0.7)
    low_gain = normalized_gain < 0.3
    
    return {
        'mean_normalized_gain': float(normalized_gain.mean()),
        'median_normalized_gain': float(normalized_gain.median()),
        'high_gain_count': int(high_gain.sum()),
        'high_gain_pct': float(high_gain.sum()/len(normalized_gain)*100),
        'medium_gain_count': int(medium_gain.sum()),
        'medium_gain_pct': float(medium_gain.sum()/len(normalized_gain)*100),
        'low_gain_count': int(low_gain.sum()),
        'low_gain_pct': float(low_gain.sum()/len(normalized_gain)*100)
    }

def generate_comprehensive_report(merged_df, pre_total_col='pre_total', post_total_col='post_total'):
    """
    Generate a comprehensive analysis report
    """
    pre_scores = merged_df[pre_total_col]
    post_scores = merged_df[post_total_col]
    
    # Basic statistics
    stats_summary = {
        'pre_mean': float(pre_scores.mean()),
        'pre_median': float(pre_scores.median()),
        'pre_std': float(pre_scores.std()),
        'post_mean': float(post_scores.mean()),
        'post_median': float(post_scores.median()),
        'post_std': float(post_scores.std()),
        'mean_gain': float((post_scores - pre_scores).mean()),
        'median_gain': float((post_scores - pre_scores).median()),
        'gain_std': float((post_scores - pre_scores).std())
    }
    
    # Statistical tests
    statistical_results = statistical_tests(pre_scores, post_scores)
    
    # Correlation analysis
    correlation_results = correlation_analysis(pre_scores, post_scores)
    
    # Normalized gain analysis
    normalized_gain_results = compute_normalized_gain(pre_scores, post_scores)
    
    # Combine all results
    comprehensive_report = {
        'descriptive_statistics': stats_summary,
        'inferential_statistics': statistical_results,
        'correlation_analysis': correlation_results,
        'normalized_gain_analysis': normalized_gain_results
    }
    
    return comprehensive_report