# xinfer module

The `xinfer` module provides a flexible and extensible framework for working with various AI models, particularly focusing on vision-language models like BLIP2.

## Main Functions

### list_models

::: xinfer.list_models

This function displays a table of all available models in the ModelRegistry. It uses the Rich library to create a formatted table with two columns: "Implementation" and "Model Type".

### get_model

::: xinfer.get_model

This function retrieves a model instance based on the specified model type and implementation. It returns an instance of a class that inherits from `BaseModel`.

Parameters:
- `model_type` (str): The type of the model to retrieve.
- `implementation` (str): The implementation of the model.
- `**kwargs`: Additional keyword arguments to pass to the model constructor.

Returns:
- An instance of the requested model.

Raises:
- `ValueError`: If the specified implementation or model type is not supported.

## Model Registry

The `ModelRegistry` class manages the registration and retrieval of models. It provides the following class methods:

### register

::: xinfer.ModelRegistry.register

This method registers a model class with the ModelRegistry.



