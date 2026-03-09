**## Project Description**

This project is an educational **single-name CDS pricer** built in Python with a **Streamlit interface**. It implements a simple reduced-form credit model where default is driven by a hazard rate that can be either constant or piecewise-constant. The tool allows users to compute key CDS quantities such as the **fair spread, present value (PV), risky PV01, and survival/default probabilities**.

The application also includes a **bootstrap module** that calibrates a hazard rate curve from market CDS spreads at standard maturities. The goal of the project is mainly pedagogical: to provide a clear and interactive way to understand how CDS pricing works and how credit risk parameters affect the valuation.