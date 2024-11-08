import xinfer


def test_list_models():
    # Should not return any errors
    xinfer.list_models()


def test_list_models_interactive():
    xinfer.list_models(interactive=True)

    xinfer.list_models(interactive=False, limit=10)
