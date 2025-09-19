# tail -5000 ascends_server.py
#!/usr/bin/env python3
# coding: utf-8
import tornado.escape
import tornado.ioloop
import tornado.web
from tornado.escape import json_decode
from tornado.escape import json_encode
import json
import tempfile
import os
import ascends as asc
import pandas as pd
import keras
from pathlib import PurePath
from tornado.options import define, options, parse_command_line

define("port", default=7777, help="run on the given port", type=int)

class ExecuteMLWithFormDataHandler(tornado.web.RequestHandler):

    def options(self):
        # Handle CORS preflight requests
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.finish()

    def post(self):
        # Set CORS headers
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

        try:
            # Parse FormData fields
            target_col = self.get_body_argument('target_col', default='y')
            input_cols = self.get_body_argument('input_cols', default='x')
            num_fold = self.get_body_argument('num_fold', default='5')
            preset = self.get_body_argument('preset', default='default')
            scaler_option = self.get_body_argument('scaler', default='AutoLoad')
            model_abbr = self.get_body_argument('model_abbr', default='RF')
            input_data_str = self.get_body_argument('input_data', default='[]')

            # Parse the JSON data from input_data field
            input_data = json.loads(input_data_str)

            # Create temporary CSV file from the input data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                # Write header
                f.write(f"{input_cols},{target_col}\n")
                # Write data rows
                for item in input_data:
                    f.write(f"{item['x']},{item['y']}\n")
                file_path = f.name

            # Process with temporary file
            data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(
                csv_file=file_path,
                input_col=[input_cols],
                cols_to_remove=[],
                target_col=target_col,
                random_state=None
            )

            # Clean up temporary file
            os.unlink(file_path)

            if(preset=='default'):
                model_parameters = asc.default_model_parameters()
            else:
                model_parameters = asc.load_model_parameter_from_file(preset)
            if scaler_option=="AutoLoad":
                scaler_option = model_parameters['scaler_option']

            # Run the model
            try:
                if model_abbr=='NET':
                    lr = float(model_parameters['net_learning_rate'])
                    layer = int(model_parameters['net_layer_n'])
                    dropout = float(model_parameters['net_dropout'])
                    l_2 = float(model_parameters['net_l_2'])
                    epochs = int(model_parameters['net_epochs'])
                    batch_size = int(model_parameters['net_batch_size'])
                    net_structure = [int(x) for x in model_parameters['net_structure'].split(" ")]

                    optimizer = keras.optimizers.Adam(lr=lr)
                    model = asc.net_define(params=net_structure, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer)
                    predictions, actual_values = asc.cross_val_predict_net(model, epochs=epochs, batch_size=batch_size, x_train = x_train, y_train = y_train, verbose = 0, scaler_option = scaler_option, force_to_proceed=True)
                    MAE, R2 = asc.evaluate(predictions, actual_values)

                else:
                    model = asc.define_model_regression(model_abbr, model_parameters, x_header_size = x_train.shape[1])
                    predictions, actual_values = asc.train_and_predict(model, x_train, y_train, scaler_option=scaler_option, num_of_folds=int(num_fold))
                    MAE, R2 = asc.evaluate(predictions, actual_values)

            except Exception as e:
                print(f"Error during model execution: {e}")
                MAE = -1
                R2 = -1

            # Build response
            response_to_send = {}
            response_to_send["MAE"]=float(MAE)
            response_to_send["R2"]=float(R2)
            response_to_send["input_cols"]=[input_cols] if isinstance(input_cols, str) else input_cols
            response_to_send["target_col"]=target_col
            response_to_send["model_abbr"]=model_abbr
            response_to_send["num_fold"]=num_fold
            response_to_send["scaler"]=scaler_option

            # Add fitting line if available
            if MAE!=-1 and 'predictions' in locals() and 'actual_values' in locals():
                # Create fitting line data for the frontend
                fitting_line = []
                for i, pred in enumerate(predictions):
                    if i < len(actual_values):
                        fitting_line.append({
                            "x": float(actual_values[i]),
                            "y": float(pred)
                        })
                response_to_send["fitting_line"] = fitting_line

            print(response_to_send)
            self.write(json.dumps(response_to_send))

        except Exception as e:
            print(f"Error in ExecuteMLWithFormDataHandler: {e}")
            import traceback
            traceback.print_exc()
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))

def main():
    print("\n * ASCENDS: Minimal ML API ")
    print(" * Web Server ver 0.1 \n")
    print(" programmed by Matt Sangkeun Lee (lees4@ornl.gov) ")
    print(" please go to : http://localhost:7777/")

    parse_command_line()
    app = tornado.web.Application(
        [
            (r"/execute_ml_with_formdata", ExecuteMLWithFormDataHandler),
        ],
        cookie_secret="cookingpapamattlee",
        xsrf_cookies=False,
    )
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()
