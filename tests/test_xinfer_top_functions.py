import xinfer


def test_list_models():
    # Should not return any errors
    xinfer.list_models()


def test_list_models_interactive():
    result = xinfer.list_models(interactive=True)
    assert result is not None

    result = xinfer.list_models(interactive=True, limit=10)
    assert len(result) == 10 + 2
