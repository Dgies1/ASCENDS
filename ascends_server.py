#!/usr/bin/env python3
# coding: utf-8
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import tempfile
import os
import ascends as asc
import pandas as pd
import keras
from pathlib import PurePath

app = FastAPI(
    title="ASCENDS ML API",
    description="Minimal ML API for model execution",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MLRequest(BaseModel):
    target_col: str = "y"
    input_cols: str = "x"
    num_fold: str = "5"
    preset: str = "default"
    scaler: str = "AutoLoad"
    model_abbr: str = "RF"
    input_data: List[Dict[str, float]] = []

class MLResponse(BaseModel):
    MAE: float
    R2: float
    input_cols: List[str]
    target_col: str
    model_abbr: str
    num_fold: str
    scaler: str
    fitting_line: Optional[List[Dict[str, float]]] = None

@app.post("/execute_ml", response_model=MLResponse)
async def execute_ml(request: MLRequest):
    try:
        # Extract parameters from request
        target_col = request.target_col
        input_cols = request.input_cols
        num_fold = request.num_fold
        preset = request.preset
        scaler_option = request.scaler
        model_abbr = request.model_abbr
        input_data = request.input_data

        # Create temporary CSV file from the input data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write(f"{input_cols},{target_col}\n")
            # Write data rows
            for item in input_data:
                f.write(f"{item['x']},{item['y']}\n")
            file_path = f.name

        # Process with temporary file
        data_df, x_train, x_test, y_train, y_test, header_x, header_y = asc.data_load_shuffle(
            csv_file=file_path,
            train_cols=[input_cols],
            cols_to_remove=[],
            target_col=target_col,
            random_state=None
        )

        # Clean up temporary file
        os.unlink(file_path)

        if preset == 'default':
            model_parameters = asc.default_model_parameters()
        else:
            model_parameters = asc.load_model_parameter_from_file(preset)
        if scaler_option == "AutoLoad":
            scaler_option = model_parameters['scaler_option']

        # Run the model
        try:
            if model_abbr == 'NET':
                lr = float(model_parameters['net_learning_rate'])
                layer = int(model_parameters['net_layer_n'])
                dropout = float(model_parameters['net_dropout'])
                l_2 = float(model_parameters['net_l_2'])
                epochs = int(model_parameters['net_epochs'])
                batch_size = int(model_parameters['net_batch_size'])
                net_structure = [int(x) for x in model_parameters['net_structure'].split(" ")]

                optimizer = keras.optimizers.Adam(lr=lr)
                model = asc.net_define(params=net_structure, layer_n=layer, input_size=x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer)
                predictions, actual_values = asc.cross_val_predict_net(model, epochs=epochs, batch_size=batch_size, x_train=x_train, y_train=y_train, verbose=0, scaler_option=scaler_option, force_to_proceed=True)
                MAE, R2 = asc.evaluate(predictions, actual_values)

            else:
                model = asc.define_model_regression(model_abbr, model_parameters, x_header_size=x_train.shape[1])
                predictions, actual_values = asc.train_and_predict(model, x_train, y_train, scaler_option=scaler_option, num_of_folds=int(num_fold))
                MAE, R2 = asc.evaluate(predictions, actual_values)

        except Exception as e:
            print(f"Error during model execution: {e}")
            MAE = -1
            R2 = -1

        # Build response
        response_data = {
            "MAE": float(MAE),
            "R2": float(R2),
            "input_cols": [input_cols] if isinstance(input_cols, str) else input_cols,
            "target_col": target_col,
            "model_abbr": model_abbr,
            "num_fold": num_fold,
            "scaler": scaler_option
        }

        # Add fitting line if available
        if MAE != -1 and 'predictions' in locals() and 'actual_values' in locals():
            # Create fitting line data for the frontend
            fitting_line = []
            for i, pred in enumerate(predictions):
                if i < len(actual_values):
                    fitting_line.append({
                        "x": float(actual_values[i]),
                        "y": float(pred)
                    })
            response_data["fitting_line"] = fitting_line

        return response_data

    except Exception as e:
        print(f"Error in execute_ml: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "ASCENDS ML API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    print("\n * ASCENDS: Minimal ML API with FastAPI")
    print(" * API Server ver 1.0.0 \n")
    print(" * FastAPI docs available at: http://localhost:7777/docs")
    print(" * ReDoc available at: http://localhost:7777/redoc")
    uvicorn.run(app, host="0.0.0.0", port=7777)
