# Down-and-Out Option Pricing with XGBoost

This project implements a hybrid approach to derivative pricing by combining **Monte Carlo simulations** with a gradient-boosted machine learning model (**XGBoost**). The goal is to approximate the fair value of a path-dependent "Down-and-Out" call option more efficiently than traditional simulation methods.

### Motivation

This project was built as a portfolio piece for junior quantitative finance and data science roles. It focuses on bridging traditional stochastic calculus (Geometric Brownian Motion) with modern supervised learning to solve complex path-dependent pricing problems.

### Methodology

* **Path Simulation**: Generated 20,000 price paths using Geometric Brownian Motion (GBM) with parameters: , , , and .
* **Path Logic**: Incorporated a "knock-out" barrier at ; if the price hits this level at any point during the 252-day period, the option expires worthless ().
* **Feature Engineering**: Extracted 7 key features from paths, including volatility, mean price, and barrier proximity (how close the minimum price got to ).
* **ML Training**: Trained an XGBoost Regressor to map these path features to the simulated "ground truth" option price.
* **Evaluation**: Assessed model accuracy using RMSE () and visualized prediction-vs-actual performance.

### Tools and Libraries

* **Python**
* **NumPy / pandas**: For the simulation engine and data structuring.
* **XGBoost**: For high-performance gradient-boosted regression.
* **scikit-learn**: For scaling, pipeline management, and error metrics.
* **matplotlib**: For visualizing stock paths and model results.

### Structure

* `daily_stock_classifier.ipynb`: Main notebook containing the simulation engine, feature engineering, and XGBoost pipeline.
* `daily_stock_classifier_model.pkl`: Serialized trained model for future inference.
* `README.md`: Project overview and methodology.

### Notes

This implementation is for educational and research purposes only. It uses simulated data to demonstrate a proof-of-concept for ML-based derivative pricing and does not constitute financial advice.

### Possible Extensions

* Integrating real-market Implied Volatility (IV) data via APIs (e.g., yfinance).
* Expanding to other path-dependent structures like Asian or Lookback options.
* Sensitivity analysis (Greeks) of the XGBoost model predictions.

**Author:** Hripsime
