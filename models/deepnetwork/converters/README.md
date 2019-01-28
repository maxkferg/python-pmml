# Converters

Converters convert models from a language-specific format (such as a Keras model) to a stabdard PMML representation. We impliment the folowwing converters in this folder:

- Keras -> PMML
- Pytorch -> PMML

From an implimentation perspective, we first convert the language-specifi model, to our own intermediate representaion `core.DeepNetwork`. The intermendiate representation can thn be written to PMML.

## Usage

See the generate_models.py for an example of how Keras models can be converted to PMML files.
In general, the conversion process works as follows:

```python
from converters.keras import convert

keras_model = keras.Model("/path/to/model")
class_map = {"0": "bird", "1": "dog", "2": "fish"}
intermediate = convert(keras_model, class_map, description="Bird Dog Fish Model"):
intermediate.save_pmml(output_path, weights_path=weights_path, save_weights=False)
```

```python
from converters.keras import convert

torch_model = torch.Model("/path/to/model")
class_map = {"0": "bird", "1": "dog", "2": "fish"}
intermediate = convert(torch_model, class_map, description="Bird Dog Fish Model"):
intermediate.save_pmml(output_path, weights_path=weights_path, save_weights=False)```