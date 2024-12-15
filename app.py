from flask import Flask, render_template, request, jsonify
import pandas as pd
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot

app = Flask(__name__)

# Supported models
MODELS = {
    "GaussianCopula": GaussianCopulaSynthesizer,
    "CTGAN": CTGANSynthesizer,
    "VAE":TVAESynthesizer,
    "Copula GAN": CopulaGANSynthesizer   
}

@app.route('/')
def index():
    return render_template('index.html', models=MODELS.keys())

@app.route('/generate', methods=['POST'])
def generate_data():
    try:
        # Retrieve form data
        file = request.files['dataset']
        model_name = request.form['model']
        n_rows = int(request.form['rows'])

        # Load the uploaded dataset
        historical_data = pd.read_csv(file)

        # Step 1: Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(historical_data)
        metadata.validate()

        # Step 2: Initialize the selected synthesizer
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' is not supported.")
        synthesizer = MODELS[model_name](metadata)

        # Step 3: Train the synthesizer
        synthesizer.fit(historical_data)

        # Step 4: Generate synthetic data
        synthetic_data = synthesizer.sample(num_rows=n_rows)
        # Extract the first 10 rows of the synthetic data
        synthetic_data_preview = synthetic_data.head(10).to_dict(orient='records')


        # Step 5: Evaluate the quality of the synthetic data
        quality_report = run_diagnostic(historical_data, synthetic_data, metadata)
        quality_report_dict = quality_report.get_details(property_name='Data Validity').to_dict(orient='records')
        print(quality_report_dict)

        # Step 6: Generate a column plot for evaluation
        column_name = 'energy_session'  # Replace with a valid column name from your dataset
        if column_name not in historical_data.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset.")
        fig = get_column_plot(
            real_data=historical_data,
            synthetic_data=synthetic_data,
            column_name=column_name,
            metadata=metadata
        )
        plot_path = "static/synthetic_data_plot.png"
        fig.write_image(plot_path)

        # Save synthetic data to return as a download
        synthetic_data_path = "static/synthetic_data.csv"
        synthetic_data.to_csv(synthetic_data_path, index=False)

        return jsonify({
            "success": True,
            "synthetic_data_preview": synthetic_data_preview,
            "quality_score": quality_report_dict,
            "download_link": f"/{synthetic_data_path}",
            "plot_link": f"/{plot_path}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
