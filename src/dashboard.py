import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from src.pipeline import AnnotationPipeline
import json

class Dashboard:
    def __init__(self):
        self.pipeline = AnnotationPipeline(output_dir="processed_data")
        
    def process_files(self, files):
        """Process uploaded files and return metrics"""
        results = []
        metrics_data = []
        
        for file in files:
            file_path = Path(file.name)
            df, text_columns = self.pipeline._read_dataset(file_path)
            
            # Process each column
            for column in text_columns:
                processed_df = self.pipeline.process_single_column(df, column)
                metrics_file = f"processed_data/processed_{file_path.stem}_metrics.json"
                
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    metrics_data.append({
                        'file': file_path.name,
                        'column': column,
                        **metrics[column]
                    })
                    
                results.append({
                    'File': file_path.name,
                    'Column': column,
                    'Rows Processed': len(df),
                    'Quality Score': metrics[column]['nlp_quality']
                })
                
        return (
            self.create_metrics_plot(metrics_data),
            pd.DataFrame(results)
        )
    
    def create_metrics_plot(self, metrics_data):
        """Create quality metrics visualization"""
        fig = go.Figure()
        
        for metric in metrics_data:
            fig.add_trace(go.Scatterpolar(
                r=[
                    metric['completeness']['null_score'],
                    metric['consistency'],
                    metric['validity'],
                    metric['uniqueness'],
                    metric['nlp_quality']
                ],
                theta=['Completeness', 'Consistency', 'Validity', 
                       'Uniqueness', 'NLP Quality'],
                name=f"{metric['file']} - {metric['column']}"
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True
        )
        
        return fig
    
    def create_interface(self):
        """Create Gradio interface"""
        # Create custom theme
        theme = gr.themes.Base(
            font=["Helvetica", "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=["Roboto Mono", "monospace"]
        )
        
        with gr.Blocks(theme=theme) as demo:
            gr.Markdown("# CleanLLM Data Preparation Pipeline")
            
            with gr.Row():
                file_input = gr.File(
                    label="Upload Dataset",
                    file_count="multiple",
                    type="filepath"  # Changed from "file" to "filepath"
                )
                
            with gr.Row():
                quality_metrics = gr.Plot(label="Quality Metrics")
                
            with gr.Row():
                results_table = gr.DataFrame(
                    label="Processing Results",
                    interactive=False  # Prevent editing
                )
                
            # Update event handler
            file_input.change(  # Use change instead of upload
                fn=self.process_files,
                inputs=[file_input],
                outputs=[quality_metrics, results_table],
                api_name="process_files"  # Explicitly name the endpoint
            )
            
        return demo

    def launch_dashboard(self):
        demo = self.create_interface()
        # Configure launch parameters
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.launch_dashboard() 