"""
Quality validation and metrics for synthetic healthcare data.
Compares synthetic data against original data to ensure quality and fidelity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
from scipy import stats
import asyncio

logger = logging.getLogger(__name__)

class QualityValidator:
    """
    Validates synthetic data quality against original data.
    Performs statistical tests and quality assessments.
    """
    
    def __init__(self):
        self.significance_level = 0.05
        self.correlation_threshold = 0.1  # Acceptable correlation difference
        self.distribution_threshold = 0.05  # P-value threshold for distribution tests
    
    async def validate_synthetic_data(
        self,
        original_data: List[Dict[str, Any]],
        synthetic_data: List[Dict[str, Any]],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate synthetic data quality against original data.
        
        Args:
            original_data: Original cleaned data
            synthetic_data: Generated synthetic data
            schema: Data schema information
            
        Returns:
            Validation report with quality metrics
        """
        
        if not original_data or not synthetic_data:
            return {
                "error": "Empty data provided for validation",
                "overall_score": 0.0,
                "passed": False
            }
        
        # Convert to DataFrames
        original_df = pd.DataFrame(original_data)
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Ensure same columns
        common_columns = list(set(original_df.columns) & set(synthetic_df.columns))
        if not common_columns:
            return {
                "error": "No common columns between original and synthetic data",
                "overall_score": 0.0,
                "passed": False
            }
        
        original_df = original_df[common_columns]
        synthetic_df = synthetic_df[common_columns]
        
        # Perform validation tests
        distribution_results = await self._test_distributions(original_df, synthetic_df, schema)
        correlation_results = await self._test_correlations(original_df, synthetic_df, schema)
        constraint_results = await self._test_constraints(synthetic_df, schema)
        privacy_results = await self._test_privacy_preservation(original_df, synthetic_df)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            distribution_results,
            correlation_results,
            constraint_results,
            privacy_results
        )
        
        # Determine if validation passed
        passed = overall_score >= 0.7  # 70% threshold
        
        validation_report = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "original_record_count": len(original_data),
            "synthetic_record_count": len(synthetic_data),
            "columns_validated": common_columns,
            "distribution_comparison": distribution_results,
            "correlation_preservation": correlation_results,
            "constraint_violations": constraint_results,
            "privacy_preservation": privacy_results,
            "overall_score": overall_score,
            "passed": passed,
            "recommendations": self._generate_recommendations(
                distribution_results, correlation_results, constraint_results
            )
        }
        
        return validation_report
    
    async def _test_distributions(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test if synthetic data preserves original distributions"""
        
        distribution_tests = {}
        
        for column in original_df.columns:
            field_info = schema.get("fields", {}).get(column, {})
            field_type = field_info.get("type", "unknown")
            
            try:
                if field_type in ["integer", "float"]:
                    # Kolmogorov-Smirnov test for continuous distributions
                    result = self._ks_test(original_df[column], synthetic_df[column])
                elif field_type == "categorical":
                    # Chi-square test for categorical distributions
                    result = self._chi_square_test(original_df[column], synthetic_df[column])
                else:
                    # Basic distribution comparison
                    result = self._basic_distribution_test(original_df[column], synthetic_df[column])
                
                distribution_tests[column] = result
            
            except Exception as e:
                logger.error(f"Distribution test failed for column {column}: {str(e)}")
                distribution_tests[column] = {
                    "test_type": "error",
                    "passed": False,
                    "error": str(e)
                }
        
        return distribution_tests
    
    def _ks_test(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for continuous distributions"""
        
        # Remove null values
        orig_clean = original.dropna()
        synth_clean = synthetic.dropna()
        
        if len(orig_clean) == 0 or len(synth_clean) == 0:
            return {
                "test_type": "ks_test",
                "statistic": None,
                "p_value": None,
                "passed": False,
                "error": "Insufficient data for test"
            }
        
        try:
            # Convert to numeric
            orig_numeric = pd.to_numeric(orig_clean, errors='coerce').dropna()
            synth_numeric = pd.to_numeric(synth_clean, errors='coerce').dropna()
            
            statistic, p_value = stats.ks_2samp(orig_numeric, synth_numeric)
            
            return {
                "test_type": "ks_test",
                "statistic": round(statistic, 4),
                "p_value": round(p_value, 4),
                "passed": p_value > self.distribution_threshold,
                "interpretation": "Distributions are similar" if p_value > self.distribution_threshold else "Distributions differ significantly"
            }
        
        except Exception as e:
            return {
                "test_type": "ks_test",
                "passed": False,
                "error": str(e)
            }
    
    def _chi_square_test(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """Chi-square test for categorical distributions"""
        
        try:
            # Get value counts
            orig_counts = original.value_counts()
            synth_counts = synthetic.value_counts()
            
            # Align categories
            all_categories = set(orig_counts.index) | set(synth_counts.index)
            
            orig_freq = [orig_counts.get(cat, 0) for cat in all_categories]
            synth_freq = [synth_counts.get(cat, 0) for cat in all_categories]
            
            # Normalize to proportions
            orig_prop = np.array(orig_freq) / sum(orig_freq)
            synth_prop = np.array(synth_freq) / sum(synth_freq)
            
            # Expected frequencies for synthetic data
            expected_freq = orig_prop * sum(synth_freq)
            
            # Chi-square test
            statistic, p_value = stats.chisquare(synth_freq, expected_freq)
            
            return {
                "test_type": "chi_square",
                "statistic": round(statistic, 4),
                "p_value": round(p_value, 4),
                "passed": p_value > self.distribution_threshold,
                "categories_compared": len(all_categories),
                "interpretation": "Distributions are similar" if p_value > self.distribution_threshold else "Distributions differ significantly"
            }
        
        except Exception as e:
            return {
                "test_type": "chi_square",
                "passed": False,
                "error": str(e)
            }
    
    def _basic_distribution_test(self, original: pd.Series, synthetic: pd.Series) -> Dict[str, Any]:
        """Basic distribution comparison for other data types"""
        
        try:
            orig_unique = set(original.dropna().astype(str))
            synth_unique = set(synthetic.dropna().astype(str))
            
            # Jaccard similarity
            intersection = len(orig_unique & synth_unique)
            union = len(orig_unique | synth_unique)
            jaccard_similarity = intersection / union if union > 0 else 0
            
            return {
                "test_type": "basic_comparison",
                "jaccard_similarity": round(jaccard_similarity, 4),
                "passed": jaccard_similarity > 0.5,
                "original_unique_values": len(orig_unique),
                "synthetic_unique_values": len(synth_unique),
                "common_values": intersection
            }
        
        except Exception as e:
            return {
                "test_type": "basic_comparison",
                "passed": False,
                "error": str(e)
            }
    
    async def _test_correlations(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test if synthetic data preserves correlations"""
        
        correlation_tests = {}
        
        # Get numeric columns
        numeric_columns = []
        for column in original_df.columns:
            field_info = schema.get("fields", {}).get(column, {})
            if field_info.get("type") in ["integer", "float"]:
                numeric_columns.append(column)
        
        if len(numeric_columns) < 2:
            return {
                "message": "Insufficient numeric columns for correlation analysis",
                "numeric_columns_count": len(numeric_columns)
            }
        
        try:
            # Calculate correlation matrices
            orig_corr = original_df[numeric_columns].corr()
            synth_corr = synthetic_df[numeric_columns].corr()
            
            # Compare correlations for each pair
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i < j:  # Avoid duplicates and self-correlation
                        pair_name = f"{col1}_vs_{col2}"
                        
                        orig_corr_val = orig_corr.loc[col1, col2]
                        synth_corr_val = synth_corr.loc[col1, col2]
                        
                        # Handle NaN correlations
                        if pd.isna(orig_corr_val) or pd.isna(synth_corr_val):
                            correlation_tests[pair_name] = {
                                "original": None,
                                "synthetic": None,
                                "difference": None,
                                "passed": False,
                                "error": "NaN correlation values"
                            }
                            continue
                        
                        difference = abs(orig_corr_val - synth_corr_val)
                        passed = difference <= self.correlation_threshold
                        
                        correlation_tests[pair_name] = {
                            "original": round(orig_corr_val, 4),
                            "synthetic": round(synth_corr_val, 4),
                            "difference": round(difference, 4),
                            "passed": passed
                        }
        
        except Exception as e:
            logger.error(f"Correlation test failed: {str(e)}")
            return {"error": str(e)}
        
        return correlation_tests
    
    async def _test_constraints(
        self,
        synthetic_df: pd.DataFrame,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test if synthetic data violates schema constraints"""
        
        violations = []
        total_constraints = 0
        
        constraints = schema.get("constraints", [])
        
        for constraint in constraints:
            total_constraints += 1
            constraint_type = constraint.get("type")
            field = constraint.get("field")
            
            if field not in synthetic_df.columns:
                violations.append({
                    "constraint": constraint,
                    "violation": f"Field {field} not found in synthetic data"
                })
                continue
            
            try:
                if constraint_type == "range":
                    min_val = constraint.get("min")
                    max_val = constraint.get("max")
                    
                    if min_val is not None:
                        below_min = (synthetic_df[field] < min_val).sum()
                        if below_min > 0:
                            violations.append({
                                "constraint": constraint,
                                "violation": f"{below_min} values below minimum {min_val}"
                            })
                    
                    if max_val is not None:
                        above_max = (synthetic_df[field] > max_val).sum()
                        if above_max > 0:
                            violations.append({
                                "constraint": constraint,
                                "violation": f"{above_max} values above maximum {max_val}"
                            })
                
                elif constraint_type == "not_null":
                    null_count = synthetic_df[field].isnull().sum()
                    if null_count > 0:
                        violations.append({
                            "constraint": constraint,
                            "violation": f"{null_count} null values in non-nullable field"
                        })
                
                elif constraint_type == "unique":
                    duplicate_count = synthetic_df[field].duplicated().sum()
                    if duplicate_count > 0:
                        violations.append({
                            "constraint": constraint,
                            "violation": f"{duplicate_count} duplicate values in unique field"
                        })
            
            except Exception as e:
                violations.append({
                    "constraint": constraint,
                    "violation": f"Error checking constraint: {str(e)}"
                })
        
        return {
            "total_constraints": total_constraints,
            "violations": violations,
            "violation_count": len(violations),
            "passed": len(violations) == 0
        }
    
    async def _test_privacy_preservation(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Test privacy preservation (no direct copying)"""
        
        privacy_tests = {}
        
        for column in original_df.columns:
            if column in synthetic_df.columns:
                try:
                    # Check for exact matches (potential privacy leak)
                    orig_values = set(original_df[column].dropna().astype(str))
                    synth_values = set(synthetic_df[column].dropna().astype(str))
                    
                    exact_matches = len(orig_values & synth_values)
                    match_percentage = (exact_matches / len(orig_values)) * 100 if len(orig_values) > 0 else 0
                    
                    # Privacy preserved if low exact match percentage
                    privacy_preserved = match_percentage < 10  # Less than 10% exact matches
                    
                    privacy_tests[column] = {
                        "exact_matches": exact_matches,
                        "match_percentage": round(match_percentage, 2),
                        "privacy_preserved": privacy_preserved,
                        "risk_level": self._assess_privacy_risk(match_percentage)
                    }
                
                except Exception as e:
                    privacy_tests[column] = {
                        "error": str(e),
                        "privacy_preserved": False
                    }
        
        return privacy_tests
    
    def _assess_privacy_risk(self, match_percentage: float) -> str:
        """Assess privacy risk based on match percentage"""
        
        if match_percentage < 5:
            return "LOW"
        elif match_percentage < 15:
            return "MEDIUM"
        elif match_percentage < 30:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _calculate_overall_score(
        self,
        distribution_results: Dict[str, Any],
        correlation_results: Dict[str, Any],
        constraint_results: Dict[str, Any],
        privacy_results: Dict[str, Any]
    ) -> float:
        """Calculate overall validation score"""
        
        scores = []
        
        # Distribution score
        if isinstance(distribution_results, dict) and "error" not in distribution_results:
            dist_passed = sum(1 for result in distribution_results.values() 
                            if isinstance(result, dict) and result.get("passed", False))
            dist_total = len(distribution_results)
            dist_score = dist_passed / dist_total if dist_total > 0 else 0
            scores.append(dist_score * 0.4)  # 40% weight
        
        # Correlation score
        if isinstance(correlation_results, dict) and "error" not in correlation_results:
            corr_passed = sum(1 for result in correlation_results.values() 
                            if isinstance(result, dict) and result.get("passed", False))
            corr_total = len(correlation_results)
            corr_score = corr_passed / corr_total if corr_total > 0 else 1
            scores.append(corr_score * 0.3)  # 30% weight
        
        # Constraint score
        if isinstance(constraint_results, dict):
            constraint_score = 1.0 if constraint_results.get("passed", False) else 0.5
            scores.append(constraint_score * 0.2)  # 20% weight
        
        # Privacy score
        if isinstance(privacy_results, dict):
            privacy_passed = sum(1 for result in privacy_results.values() 
                               if isinstance(result, dict) and result.get("privacy_preserved", False))
            privacy_total = len(privacy_results)
            privacy_score = privacy_passed / privacy_total if privacy_total > 0 else 1
            scores.append(privacy_score * 0.1)  # 10% weight
        
        return round(sum(scores), 3) if scores else 0.0
    
    def _generate_recommendations(
        self,
        distribution_results: Dict[str, Any],
        correlation_results: Dict[str, Any],
        constraint_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving synthetic data quality"""
        
        recommendations = []
        
        # Distribution recommendations
        if isinstance(distribution_results, dict):
            failed_distributions = [col for col, result in distribution_results.items() 
                                  if isinstance(result, dict) and not result.get("passed", True)]
            
            if failed_distributions:
                recommendations.append(
                    f"Consider improving distribution matching for columns: {', '.join(failed_distributions[:3])}"
                )
        
        # Correlation recommendations
        if isinstance(correlation_results, dict):
            failed_correlations = [pair for pair, result in correlation_results.items() 
                                 if isinstance(result, dict) and not result.get("passed", True)]
            
            if failed_correlations:
                recommendations.append(
                    "Consider using correlation-preserving synthesis methods"
                )
        
        # Constraint recommendations
        if isinstance(constraint_results, dict) and constraint_results.get("violation_count", 0) > 0:
            recommendations.append(
                "Review and enforce schema constraints during synthesis"
            )
        
        if not recommendations:
            recommendations.append("Synthetic data quality is good. No major improvements needed.")
        
        return recommendations
