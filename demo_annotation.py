import json
from rich.console import Console
from rich.table import Table
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import argparse
from src.annotation_system import AnnotationSystem, AnnotationConfig

class AnnotationDemo:
    def __init__(self):
        self.console = Console()
        self.config = AnnotationConfig(
            enable_entities=True,
            enable_sentiment=True,
            enable_topics=True,
            enable_keywords=True,
            enable_language=True,
            enable_instruction=True,
            min_keyword_score=0.3,
            max_keywords=5
        )
        self.system = AnnotationSystem(self.config)

    def display_annotations(self, annotations: Dict[str, Any]):
        """Display annotations in a formatted way"""
        self.console.print("\n[bold cyan]Annotation Results:[/bold cyan]")

        # Display Entities
        if 'entities' in annotations and annotations['entities']:
            table = Table(title="Named Entities")
            table.add_column("Text", style="cyan")
            table.add_column("Label", style="green")
            table.add_column("Position", style="yellow")

            for entity in annotations['entities']:
                table.add_row(
                    entity['text'],
                    entity['label'],
                    f"{entity['start']}-{entity['end']}"
                )
            self.console.print(table)

        # Display Sentiment
        if 'sentiment' in annotations:
            sentiment = annotations['sentiment']
            self.console.print("\n[bold]Sentiment Analysis:[/bold]")
            self.console.print(f"Label: {sentiment['label']}")
            self.console.print(f"Score: {sentiment['score']:.3f}")

        # Display Topics
        if 'topics' in annotations and annotations['topics']:
            self.console.print("\n[bold]Main Topics:[/bold]")
            for topic in annotations['topics']:
                self.console.print(f"• {topic}")

        # Display Keywords
        if 'keywords' in annotations and annotations['keywords']:
            table = Table(title="Keywords")
            table.add_column("Keyword", style="cyan")
            table.add_column("POS", style="green")
            table.add_column("Score", style="yellow")

            for keyword in annotations['keywords']:
                table.add_row(
                    keyword['text'],
                    keyword['pos'],
                    f"{keyword['score']:.3f}"
                )
            self.console.print(table)

        # Display Language
        if 'language' in annotations:
            lang = annotations['language']
            self.console.print("\n[bold]Language Detection:[/bold]")
            self.console.print(f"Language: {lang['language']}")
            self.console.print(f"Confidence: {lang['confidence']:.2f}")

        # Display Instruction Analysis
        if 'instruction' in annotations:
            inst = annotations['instruction']
            self.console.print("\n[bold]Instruction Analysis:[/bold]")
            self.console.print(f"Is Instruction: {inst['is_instruction']}")
            if inst['command_words']:
                self.console.print(f"Command Words: {', '.join(inst['command_words'])}")

    def process_single_text(self, text: str):
        """Process and display annotations for a single text"""
        self.console.print(f"\n[bold]Input Text:[/bold] {text}")
        annotations = self.system.annotate_text(text)
        self.display_annotations(annotations)

    def process_file(self, file_path: str, output_dir: str = "annotated_output"):
        """Process texts from a file and save annotations"""
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Read file
            file_path = Path(file_path)
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")

            # Process each text column
            for column in df.select_dtypes(include=['object']).columns:
                self.console.print(f"\n[bold]Processing column:[/bold] {column}")
                
                # Process texts
                annotations = []
                with self.console.status("[bold green]Processing texts...") as status:
                    for text in df[column]:
                        if isinstance(text, str):
                            result = self.system.annotate_text(text)
                            annotations.append(result)
                        else:
                            annotations.append({})

                # Add annotations to dataframe
                df[f"{column}_annotations"] = annotations

            # Save results
            output_file = output_path / f"annotated_{file_path.name}"
            if file_path.suffix == '.csv':
                df.to_csv(output_file, index=False)
            else:
                df.to_json(output_file, orient='records')

            self.console.print(f"\n[bold green]Results saved to:[/bold green] {output_file}")

        except Exception as e:
            self.console.print(f"[bold red]Error processing file:[/bold red] {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Annotation System Demo")
    parser.add_argument("--text", help="Single text to annotate")
    parser.add_argument("--file", help="File to process (CSV or JSON)")
    parser.add_argument("--output", default="annotated_output", help="Output directory for file processing")

    args = parser.parse_args()
    demo = AnnotationDemo()

    if args.text:
        demo.process_single_text(args.text)
    elif args.file:
        demo.process_file(args.file, args.output)
    else:
        # Demo with example texts
        example_texts = [
            "Apple Inc. is planning to release a new iPhone next year. The company's CEO Tim Cook is very excited about it.",
            "Can you explain how artificial intelligence works in simple terms?",
            "This product is absolutely amazing! I love all its features and the customer service is outstanding.",
            "La Torre Eiffel es el monumento más famoso de París.",  # Test language detection
        ]

        for text in example_texts:
            demo.process_single_text(text)

if __name__ == "__main__":
    main()