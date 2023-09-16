# Crop Yield Estimation with Machine Learning

<p align="center">
    <img src="images/crop_yield.jpeg" width="400">
</p>


## Project Summary

Crop yield estimation plays a pivotal role in agricultural decision-making, impacting both farmers and the broader agricultural industry. Leveraging the power of artificial intelligence and remote sensing data, our project, "Crop Yield Estimation," focuses on predicting cotton crop yields in the field. We employ various machine learning techniques, including Multi-layer Perceptron (MLP), Random Forest (RF), Long Short-Term Memory (LSTM), and XGBoost, to provide accurate yield predictions.

### Motivation

Accurate crop yield predictions are essential for optimizing resource allocation, crop management, and ensuring food security. By utilizing data from satellite imagery and ground sensors, we aim to enhance the precision of cotton yield forecasts. Our project contributes to sustainable agriculture by providing timely and data-driven insights to farmers and stakeholders.

### Data Sources

We gather a diverse set of data sources, including:

- Enhanced Vegetation Index (EVI)
- Leaf Area Index (LAI)
- Fraction of Photosynthetically Active Radiation (FPAR) from satellite imagery
- Ground-based sensors providing data on:
  - Vapor pressure
  - Surface soil moisture
  - Subsurface soil moisture
  - Maximum temperature
  - Minimum temperature
  - Solar radiation
  - Precipitation
  - Cotton yield

These data sources collectively form the basis for our machine learning models.

### Machine Learning Methods

We explore the following machine learning methods for crop yield prediction:

1. **Multi-layer Perceptron (MLP)**: A feedforward neural network model.
2. **Random Forest (RF)**: An ensemble learning technique based on decision trees.
3. **Long Short-Term Memory (LSTM)**: A type of recurrent neural network (RNN) suitable for time-series data.
4. **XGBoost**: A gradient boosting algorithm known for its efficiency and accuracy.

### Evaluation Metrics

To assess the performance of our models, we employ the following evaluation metrics:

- **Root Mean Square Error (RMSE)**: A measure of the model's prediction error.
- **Coefficient of Determination (𝑅²)**: Indicates the proportion of the variance in the dependent variable that is predictable.

## Getting Started

To reproduce our results or use our models, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/abalmumcu/crop-yield-estimation.git
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Explore the Jupyter notebooks in the `notebooks` directory for data preprocessing, model training, and evaluation.

4. Customize and adapt the models to your specific use case.

## Results

We present the results of our crop yield estimation models in the `results` directory. You can find detailed analyses and visualizations of our predictions there.

## Contributing

We welcome contributions and collaboration from the community. If you have ideas for improvements or would like to contribute to this project, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.