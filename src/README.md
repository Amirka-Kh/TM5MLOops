# Extract, Process, Validate, Train

In this directory we have code to: 
- extract data from source,
- sample this data by params in hydra,
- version data with dvc and git tool,
- transform data (one-hot encoding, filling Null values),
- check and validate data with Great Expectation tool,
- store transformed data (features) as ZenML artifacts,
- train model in parallel with logging experiment data (mlflow).

## Environment 

Set up an environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r mlflow.requirements.txt
```

After preparing the environment you should add such parameters to `.bashrc`:
    
```
# REPLACE <project-folder-path> with your project folder path
cd <project-folder-path>
echo "export ZENML_CONFIG_PATH=$PWD/services/zenml" >> ~/.bashrc
echo "export PYTHONPATH=$PWD" >> ~/.bashrc

# Run the file
source ~/.bashrc
```
    
## Extract Data

Before running experiments, we need to have data. Thus, in first step
it is important to extract and sample data, and not forget to version it:

```azure

# To extract and sample data; validate its quality
python src/data.py

# To version data (we assume you logged to your github account)
python src/version_data.py
```

## Transform and Load Data

In order to use data for training, we should transform data to convenient
format and store it comfortable place.

Run zenml pipeline `python src/prepare_data.py`, it will transform data and put it to feature store.
We will use zenml api then to fetch processed data.

## Train

Run steps previous steps one more time to have two different data features in datastore.
We will need them in mlflow experiment.

After that you should have two `features_target` artifacts you can check them by
`zenml artifact version list --name features_target`. Check if `tag` number is similar to one in
`config/data_version.yaml`, also if there are tags lower than number in data_version.yaml.
```
┠──────────────────────────────┼─────────────────┼─────────┼──────────────────────────────┼──────────────┼──────────────────────────────┼───────────────────────────────┼──────────┨
┃ ad968501-eaf2-417c-865a-7613 │ features_target │ 4       │ /mnt/c/Users/amira/PycharmPr │ DataArtifact │ module='zenml.materializers. │ module='pandas.core.frame'    │ ['1.18'] ┃
┃           f4bc245e           │                 │         │ ojects/MLOps/services/zenml/ │              │ pandas_materializer'         │ attribute='DataFrame'         │          ┃
┃                              │                 │         │ local_stores/c028da59-655d-4 │              │ attribute='PandasMaterialize │ type=<SourceType.DISTRIBUTION │          ┃
┃                              │                 │         │ 70f-9e3c-d943d6fe3ad6/custom │              │ r'                           │ _PACKAGE:                     │          ┃
┃                              │                 │         │ _artifacts/features_target/0 │              │ type=<SourceType.INTERNAL:   │ 'distribution_package'>       │          ┃
┃                              │                 │         │ f11a5b2-161e-4299-b025-a971d │              │ 'internal'>                  │ package_name='pandas'         │          ┃
┃                              │                 │         │ 67375e9                      │              │                              │ version='2.2.2'               │          ┃
┠──────────────────────────────┼─────────────────┼─────────┼──────────────────────────────┼──────────────┼──────────────────────────────┼───────────────────────────────┼──────────┨
┃ ea4625b3-f493-4e73-b73b-0955 │ features_target │ 5       │ /mnt/c/Users/amira/PycharmPr │ DataArtifact │ module='zenml.materializers. │ module='pandas.core.frame'    │ ['1.19'] ┃
┃           68bf6b0b           │                 │         │ ojects/MLOps/services/zenml/ │              │ pandas_materializer'         │ attribute='DataFrame'         │          ┃
┃                              │                 │         │ local_stores/c028da59-655d-4 │              │ attribute='PandasMaterialize │ type=<SourceType.DISTRIBUTION │          ┃
┃                              │                 │         │ 70f-9e3c-d943d6fe3ad6/custom │              │ r'                           │ _PACKAGE:                     │          ┃
┃                              │                 │         │ _artifacts/features_target/f │              │ type=<SourceType.INTERNAL:   │ 'distribution_package'>       │          ┃
┃                              │                 │         │ 1a69fb3-e581-4e15-b661-eed66 │              │ 'internal'>                  │ package_name='pandas'         │          ┃
┃                              │                 │         │ f4eaec7                      │              │                              │ version='2.2.2'               │          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━┷━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┛
```
   
After these steps mlflow will be ready for runs.
Run mlflow `mlflow run . --env-manager=local`

## Giskard Fail

We tried to run giskard, tried several techniques, however encountered many issues:
```azure
$ python src/validate.py
Your 'pandas.DataFrame' is successfully wrapped by Giskard's 'Dataset' wrapper class.
2024/07/28 22:49:25 WARNING mlflow.utils.requirements_utils: Detected one or more mismatches between the model's dependencies and the current Python environment:
 - scipy (current: 1.11.0, required: scipy==1.14.0)
To fix the mismatches, call `mlflow.pyfunc.get_model_dependencies(model_uri)` to fetch the model's environment and install dependencies using the resulting environment file.
sad
[ 9.89924607 10.33406016 10.11103263  9.05889454  9.06012143]
Your 'prediction_function' is successfully wrapped by Giskard's 'PredictionFunctionModel' wrapper class.
Casting dataframe columns from {'id': 'int64', 'property_type': 'object', 'room_type': 'object', 'amenities': 'object', 'accommodates': 'int64', 'bathrooms': 'float64', 'bed_type': 'object', 'cancellation_policy': 'object', 'cleaning_fee': 'bool', 'city': 'object', 'description': 'object', 'first_review': 'object', 'host_has_profile_pic': 'object', 'host_identity_verified': 'object', 'host_response_rate': 'object', 'host_since': 'object', 'instant_bookable': 'object', 'last_review': 'object', 'latitude': 'float64', 'longitude': 'float64', 'name': 'object', 'neighbourhood': 'object', 'number_of_reviews': 'int64', 'review_scores_rating': 'float64', 'thumbnail_url': 'object', 'zipcode': 'object', 'bedrooms': 'float64', 'beds': 'float64'} to {'id': 'int64', 'property_type': 'object', 'room_type': 'object', 'amenities': 'object', 'accommodates': 'int64', 'bathrooms': 'float64', 'bed_type': 'object', 'cancellation_policy': 'object', 'cleaning_fee': 'bool', 'city': 'object', 'description': 'object', 'first_review': 'object', 'host_has_profile_pic': 'object', 'host_identi
ty_verified': 'object', 'host_response_rate': 'object', 'host_since': 'object', 'instant_bookable': 'object', 'last_review': 'object', 'latitude': 'float64', 'longitude': 'float64'
, 'name': 'object', 'neighbourhood': 'object', 'number_of_reviews': 'int64', 'review_scores_rating': 'float64', 'thumbnail_url': 'object', 'zipcode': 'object', 'bedrooms': 'float64', 'beds': 'float64'}
Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(1, '[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:2546)')': /track
Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(1, '[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:2546)')': /track
Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(1, '[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:2546)')': /api/4506789759025152/envelope/
╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/core/model_va │
│ lidation.py:88 in validate_model_execution                                                       │
│                                                                                                  │
│    85 │   │   "'prediction_function(df[feature_names].head())' does not return an error messag   │
│    86 │   )                                                                                      │
│    87 │   try:                                                                                   │
│ ❱  88 │   │   prediction = model.predict(validation_ds)                                          │
│    89 │   except Exception as e:                                                                 │
│    90 │   │   features = model.feature_names if model.feature_names is not None else validatio   │
│    91 │   │   number_of_features = len(features)                                                 │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/models/base/m │
│ odel.py:380 in predict                                                                           │
│                                                                                                  │
│   377 │   │   timer = Timer()                                                                    │
│   378 │   │                                                                                      │
│   379 │   │   if get_cache_enabled():                                                            │
│ ❱ 380 │   │   │   raw_prediction = self._predict_from_cache(dataset)                             │
│   381 │   │   else:                                                                              │
│   382 │   │   │   raw_prediction = self.predict_df(                                              │
│   383 │   │   │   │   self.prepare_dataframe(dataset.df, column_dtypes=dataset.column_dtypes,    │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/models/base/m │
│ odel.py:436 in _predict_from_cache                                                               │
│                                                                                                  │
│   433 │   │   if len(unpredicted_df) > 0:                                                        │
│   434 │   │   │   raw_prediction = self.predict_df(unpredicted_df)                               │
│   435 │   │   │   self._cache.set_cache(dataset.row_hashes[missing], raw_prediction)             │
│ ❱ 436 │   │   │   cached_predictions.loc[missing] = raw_prediction.tolist()                      │
│   437 │   │                                                                                      │
│   438 │   │   # TODO: check if there is a better solution                                        │
│   439 │   │   return np.array(np.array(cached_predictions).tolist())                             │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pandas/core/indexing. │
│ py:911 in __setitem__                                                                            │
│                                                                                                  │
│    908 │   │   self._has_valid_setitem_indexer(key)                                              │
│    909 │   │                                                                                     │
│    910 │   │   iloc = self if self.name == "iloc" else self.obj.iloc                             │
│ ❱  911 │   │   iloc._setitem_with_indexer(indexer, value, self.name)                             │
│    912 │                                                                                         │
│    913 │   def _validate_key(self, key, axis: AxisInt):                                          │
│    914 │   │   """                                                                               │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pandas/core/indexing. │
│ py:1944 in _setitem_with_indexer                                                                 │
│                                                                                                  │
│   1941 │   │   │   # We have to operate column-wise                                              │
│   1942 │   │   │   self._setitem_with_indexer_split_path(indexer, value, name)                   │
│   1943 │   │   else:                                                                             │
│ ❱ 1944 │   │   │   self._setitem_single_block(indexer, value, name)                              │
│   1945 │                                                                                         │
│   1946 │   def _setitem_with_indexer_split_path(self, indexer, value, name: str):                │
│   1947 │   │   """                                                                               │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pandas/core/indexing. │
│ py:2218 in _setitem_single_block                                                                 │
│                                                                                                  │
│   2215 │   │   self.obj._check_is_chained_assignment_possible()                                  │
│   2216 │   │                                                                                     │
│   2217 │   │   # actually do the set                                                             │
│ ❱ 2218 │   │   self.obj._mgr = self.obj._mgr.setitem(indexer=indexer, value=value)               │
│   2219 │   │   self.obj._maybe_update_cacher(clear=True, inplace=True)                           │
│   2220 │                                                                                         │
│   2221 │   def _setitem_with_indexer_missing(self, indexer, value):                              │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pandas/core/internals │
│ /managers.py:415 in setitem                                                                      │
│                                                                                                  │
│    412 │   │   │   # manager                                                                     │
│    413 │   │   │   self = self.copy()                                                            │
│    414 │   │                                                                                     │
│ ❱  415 │   │   return self.apply("setitem", indexer=indexer, value=value)                        │
│    416 │                                                                                         │
│    417 │   def diff(self, n: int) -> Self:                                                       │
│    418 │   │   # only reached with self.ndim == 2                                                │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pandas/core/internals │
│ /managers.py:363 in apply                                                                        │
│                                                                                                  │
│    360 │   │   │   if callable(f):                                                               │
│    361 │   │   │   │   applied = b.apply(f, **kwargs)                                            │
│    362 │   │   │   else:                                                                         │
│ ❱  363 │   │   │   │   applied = getattr(b, f)(**kwargs)                                         │
│    364 │   │   │   result_blocks = extend_blocks(applied, result_blocks)                         │
│    365 │   │                                                                                     │
│    366 │   │   out = type(self).from_blocks(result_blocks, self.axes)                            │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pandas/core/internals │
│ /blocks.py:1403 in setitem                                                                       │
│                                                                                                  │
│   1400 │   │   │   values = values.T                                                             │
│   1401 │   │                                                                                     │
│   1402 │   │   # length checking                                                                 │
│ ❱ 1403 │   │   check_setitem_lengths(indexer, value, values)                                     │
│   1404 │   │                                                                                     │
│   1405 │   │   if self.dtype != _dtype_obj:                                                      │
│   1406 │   │   │   # GH48933: extract_array would convert a pd.Series value to np.ndarray        │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pandas/core/indexers/ │
│ utils.py:166 in check_setitem_lengths                                                            │
│                                                                                                  │
│   163 │   │   │   │   │   and indexer.dtype == np.bool_                                          │
│   164 │   │   │   │   │   and indexer.sum() == len(value)                                        │
│   165 │   │   │   │   ):                                                                         │
│ ❱ 166 │   │   │   │   │   raise ValueError(                                                      │
│   167 │   │   │   │   │   │   "cannot set using a list-like indexer "                            │
│   168 │   │   │   │   │   │   "with a different length than the value"                           │
│   169 │   │   │   │   │   )                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
ValueError: cannot set using a list-like indexer with a different length than the value

The above exception was the direct cause of the following exception:

╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ /mnt/c/Users/amira/PycharmProjects/MLOps/src/validate.py:137 in <module>                         │
│                                                                                                  │
│   134                                                                                            │
│   135                                                                                            │
│   136 # Now, run the validation again to check for the corrected behavior                        │
│ ❱ 137 scan_results = giskard.scan(giskard_model, giskard_dataset)                                │
│   138                                                                                            │
│   139 # Save the results in `html` file                                                          │
│   140 scan_results_path = f"reports/validation_results_{model_name}_{model_version}_{dataset_n   │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/scanner/__ini │
│ t__.py:64 in scan                                                                                │
│                                                                                                  │
│   61 │   │   A scan report object containing the results of the scan.                            │
│   62 │   """                                                                                     │
│   63 │   scanner = Scanner(params, only=only)                                                    │
│ ❱ 64 │   return scanner.analyze(                                                                 │
│   65 │   │   model, dataset=dataset, features=features, verbose=verbose, raise_exceptions=rai    │
│   66 │   )                                                                                       │
│   67                                                                                             │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/scanner/scann │
│ er.py:100 in analyze                                                                             │
│                                                                                                  │
│    97 │   │   """                                                                                │
│    98 │   │   with TemporaryRootLogLevel(logging.INFO if verbose else logging.NOTSET):           │
│    99 │   │   │   # Check that the model and dataset were appropriately wrapped with Giskard     │
│ ❱ 100 │   │   │   model, dataset, model_validation_time = self._validate_model_and_dataset(mod   │
│   101 │   │   │                                                                                  │
│   102 │   │   │   # Check that provided features are valid                                       │
│   103 │   │   │   features = self._validate_features(features, model, dataset)                   │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/scanner/scann │
│ er.py:214 in _validate_model_and_dataset                                                         │
│                                                                                                  │
│   211 │   │                                                                                      │
│   212 │   │   if not model.is_text_generation:                                                   │
│   213 │   │   │   time_start = perf_counter()                                                    │
│ ❱ 214 │   │   │   validate_model(model=model, validate_ds=dataset)                               │
│   215 │   │   │   model_validation_time = perf_counter() - time_start                            │
│   216 │   │   else:                                                                              │
│   217 │   │   │   model_validation_time = None                                                   │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pydantic/validate_cal │
│ l_decorator.py:59 in wrapper_function                                                            │
│                                                                                                  │
│   56 │   │                                                                                       │
│   57 │   │   @functools.wraps(function)                                                          │
│   58 │   │   def wrapper_function(*args, **kwargs):                                              │
│ ❱ 59 │   │   │   return validate_call_wrapper(*args, **kwargs)                                   │
│   60 │   │                                                                                       │
│   61 │   │   wrapper_function.raw_function = function  # type: ignore                            │
│   62                                                                                             │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pydantic/_internal/_v │
│ alidate_call.py:81 in __call__                                                                   │
│                                                                                                  │
│   78 │   │   │   self.__return_pydantic_validator__ = None                                       │
│   79 │                                                                                           │
│   80 │   def __call__(self, *args: Any, **kwargs: Any) -> Any:                                   │
│ ❱ 81 │   │   res = self.__pydantic_validator__.validate_python(pydantic_core.ArgsKwargs(args,    │
│   82 │   │   if self.__return_pydantic_validator__:                                              │
│   83 │   │   │   return self.__return_pydantic_validator__(res)                                  │
│   84 │   │   return res                                                                          │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/core/model_va │
│ lidation.py:26 in validate_model                                                                 │
│                                                                                                  │
│    23 │   │   _do_validate_model(model, validate_ds)                                             │
│    24 │   except (ValueError, TypeError) as err:                                                 │
│    25 │   │   _track_validation_error(err, model, validate_ds)                                   │
│ ❱  26 │   │   raise err                                                                          │
│    27 │                                                                                          │
│    28 │   # TODO: switch to logger                                                               │
│    29 │   if print_validation_message:                                                           │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/core/model_va │
│ lidation.py:23 in validate_model                                                                 │
│                                                                                                  │
│    20 @configured_validate_arguments                                                             │
│    21 def validate_model(model: BaseModel, validate_ds: Optional[Dataset] = None, print_valida   │
│    22 │   try:                                                                                   │
│ ❱  23 │   │   _do_validate_model(model, validate_ds)                                             │
│    24 │   except (ValueError, TypeError) as err:                                                 │
│    25 │   │   _track_validation_error(err, model, validate_ds)                                   │
│    26 │   │   raise err                                                                          │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/core/model_va │
│ lidation.py:64 in _do_validate_model                                                             │
│                                                                                                  │
│    61 │   │   validate_features(feature_names=model.feature_names, validate_df=validate_ds.df)   │
│    62 │   │                                                                                      │
│    63 │   │   if model.is_regression:                                                            │
│ ❱  64 │   │   │   validate_model_execution(model, validate_ds)                                   │
│    65 │   │   elif model.is_text_generation:                                                     │
│    66 │   │   │   validate_model_execution(model, validate_ds, False)                            │
│    67 │   │   elif model.is_classification and validate_ds.target is not None:                   │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pydantic/validate_cal │
│ l_decorator.py:59 in wrapper_function                                                            │
│                                                                                                  │
│   56 │   │                                                                                       │
│   57 │   │   @functools.wraps(function)                                                          │
│   58 │   │   def wrapper_function(*args, **kwargs):                                              │
│ ❱ 59 │   │   │   return validate_call_wrapper(*args, **kwargs)                                   │
│   60 │   │                                                                                       │
│   61 │   │   wrapper_function.raw_function = function  # type: ignore                            │
│   62                                                                                             │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/pydantic/_internal/_v │
│ alidate_call.py:81 in __call__                                                                   │
│                                                                                                  │
│   78 │   │   │   self.__return_pydantic_validator__ = None                                       │
│   79 │                                                                                           │
│   80 │   def __call__(self, *args: Any, **kwargs: Any) -> Any:                                   │
│ ❱ 81 │   │   res = self.__pydantic_validator__.validate_python(pydantic_core.ArgsKwargs(args,    │
│   82 │   │   if self.__return_pydantic_validator__:                                              │
│   83 │   │   │   return self.__return_pydantic_validator__(res)                                  │
│   84 │   │   return res                                                                          │
│                                                                                                  │
│ /mnt/c/Users/amira/PycharmProjects/MLOps/venv/lib/python3.11/site-packages/giskard/core/model_va │
│ lidation.py:103 in validate_model_execution                                                      │
│                                                                                                  │
│   100 │   │   )                                                                                  │
│   101 │   │                                                                                      │
│   102 │   │   if not one_dimension_case:                                                         │
 (_ssl.c:2546)')': /track

```
