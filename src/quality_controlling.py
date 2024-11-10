import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from src.quality_control.metrics import QualityController, QualityMetrics, DataQualityDimension

class QualityControlling:
    def __init__(self, config_path: Optional[str] = None):
        self.console = Console()
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.controller = QualityController(self.config)
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('quality_control.log')
        
        # Create formatters and add to handlers
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        c_format = logging.Formatter(format_str)
        f_format = logging.Formatter(format_str)
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'min_completeness': 0.8,
            'min_consistency': 0.7,
            'min_validity': 0.8,
            'min_uniqueness': 0.9,
            'max_data_age_days': 30,
            'min_accuracy': 0.8,
            'text_length_threshold': 10,
            'min_overall_score': 0.75,
            'output_dir': 'quality_reports'
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                return {**default_config, **loaded_config}
            except Exception as e:
                self.logger.warning(f"Error loading config file: {e}. Using defaults.")
        
        return default_config
        
    def analyze_dataset(self, dataset_path: str) -> Tuple[bool, List[str], QualityMetrics]:
        """Analyze a single dataset"""
        try:
            # Load dataset
            df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_json(dataset_path)
            
            # Validate data
            passed, issues, metrics = self.controller.validate_data(df)
            
            # Generate detailed report
            self._generate_report(dataset_path, df, passed, issues, metrics)
            
            return passed, issues, metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing dataset {dataset_path}: {e}")
            raise
            
    def _generate_report(self, dataset_path: str, df: pd.DataFrame, 
                        passed: bool, issues: List[str], metrics: QualityMetrics):
        """Generate detailed quality report"""
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Create report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_base = Path(dataset_path).stem
        report_path = output_dir / f"quality_report_{report_base}_{timestamp}"
        
        # Generate JSON report
        report_data = {
            'dataset': dataset_path,
            'timestamp': timestamp,
            'passed': passed,
            'issues': issues,
            'metrics': metrics.to_dict(),
            'overall_score': metrics.overall_score,
            'dataset_stats': {
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'memory_usage': int(df.memory_usage(deep=True).sum()),
                'column_types': df.dtypes.astype(str).to_dict()
            }
        }
        
        with open(f"{report_path}.json", 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate human-readable report
        self._generate_readable_report(report_path, report_data)
            
    def _generate_readable_report(self, report_path: Path, report_data: Dict):
        """Generate human-readable report"""
        with open(f"{report_path}.txt", 'w') as f:
            f.write("=== Data Quality Report ===\n\n")
            f.write(f"Dataset: {report_data['dataset']}\n")
            f.write(f"Timestamp: {report_data['timestamp']}\n")
            f.write(f"Overall Status: {'PASSED' if report_data['passed'] else 'FAILED'}\n")
            f.write(f"Overall Score: {report_data['overall_score']:.2f}\n\n")
            
            f.write("--- Quality Metrics ---\n")
            for metric, score in report_data['metrics'].items():
                f.write(f"{metric.capitalize()}: {score:.2f}\n")
            
            if report_data['issues']:
                f.write("\n--- Issues Found ---\n")
                for issue in report_data['issues']:
                    f.write(f"- {issue}\n")
            
            f.write("\n--- Dataset Statistics ---\n")
            # Ensure 'stats' is defined correctly
            stats = report_data['dataset_stats']  # This line ensures 'stats' is defined
            f.write(f"Rows: {stats['rows']}\n")
            f.write(f"Columns: {stats['columns']}\n")
            f.write(f"Memory Usage: {stats['memory_usage']} bytes\n")
            
    def display_results(self, dataset_path: str, passed: bool, 
                       issues: List[str], metrics: QualityMetrics):
        """Display results in a rich formatted table"""
        # Create metrics table
        metrics_table = Table(title="Quality Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", justify="right", style="green")
        
        for metric, score in metrics.to_dict().items():
            metrics_table.add_row(
                metric.capitalize(),
                f"{score:.2f}"
            )
        
        # Add overall score
        metrics_table.add_row(
            "Overall Score",
            f"{metrics.overall_score:.2f}",
            style="bold"
        )
        
        # Display results
        self.console.print("\n=== Quality Control Results ===")
        self.console.print(f"Dataset: {dataset_path}")
        self.console.print(f"Status: ", end="")
        self.console.print("PASSED", style="green bold") if passed else self.console.print("FAILED", style="red bold")
        
        self.console.print("\nDetailed Metrics:")
        self.console.print(metrics_table)
        
        if issues:
            self.console.print("\nIssues Found:", style="yellow")
            for issue in issues:
                self.console.print(f"- {issue}")

def main():
    parser = argparse.ArgumentParser(description='Data Quality Control Tool')
    parser.add_argument('dataset', help='Path to the dataset file (CSV or JSON)')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', help='Output directory for reports')
    
    args = parser.parse_args()
    
    try:
        # Initialize quality control
        qc = QualityControlling(args.config)
        
        # Update output directory if specified
        if args.output:
            qc.config['output_dir'] = args.output
        
        # Analyze dataset
        with Progress() as progress:
            task = progress.add_task("[cyan]Analyzing dataset...", total=100)
            
            passed, issues, metrics = qc.analyze_dataset(args.dataset)
            
            progress.update(task, completed=100)
        
        # Display results
        qc.display_results(args.dataset, passed, issues, metrics)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()