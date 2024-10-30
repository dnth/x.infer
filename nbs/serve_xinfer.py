import xinfer

model = xinfer.create_model("vikhyatk/moondream2", device="cuda", dtype="float16")
model.serve()
