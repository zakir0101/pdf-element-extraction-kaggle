import os
from os.path import sep
import fitz
from PIL import Image
from flask import Flask, request, jsonify, send_file, Response
from zipfile import ZipFile
import json
from io import BytesIO
from pyngrok import ngrok
import traceback
from kaggle_secrets import UserSecretsClient

from magic_pdf.data.data_reader_writer import (
    FileBasedDataWriter,
)
from magic_pdf.data.dataset import ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model import (
    doc_analyze,
    batch_doc_analyze,
)
from magic_pdf.operators.models import InferenceResult, PipeResult


# --- 1. SETUP THE TUNNEL ---
# Authenticate ngrok using the secret you stored
user_secrets = UserSecretsClient()
ngrok.set_auth_token(user_secrets.get_secret("NGROK_AUTH_TOKEN"))

public_url = ngrok.connect(5000)
print(f"✅ Kaggle is now live at: {public_url}")
print(public_url)


def detect_layout_miner_u_online(img_bytes: bytes, data: dict):

    exam = data.get("exam", "")
    d_mode = data.get("display-mode", "")
    nr = "nr" + str(data.get("number", 0)) + "_"
    want = data.get("want", "md_content.md")

    key = exam + d_mode
    f_dir = sep.join([".", "results", key])
    os.makedirs(f_dir, exist_ok=True)
    f_path = f"{f_dir}{sep}{nr}{want}"

    md_image_dir = f"{f_dir}{sep}{nr}images"
    if not os.path.exists(f_path):
        print("ocring ....")
        image_dir = str(os.path.basename(md_image_dir))

        os.makedirs(md_image_dir, exist_ok=True)

        image_writer, md_writer = FileBasedDataWriter(
            md_image_dir
        ), FileBasedDataWriter(f_dir)

        lang = "ch_server"
        ds = ImageDataset(img_bytes, lang=lang)

        inf_res: InferenceResult = ds.apply(
            doc_analyze,
            ocr=True,
            lang=lang,
            show_log=True,
        )

        p5 = f"{f_dir}{sep}{nr}draw5.png"
        inf_res.draw_model(p5)
        pdf_to_png(p5)

        pip_res: PipeResult = inf_res.pipe_ocr_mode(image_writer, lang=lang)

        pip_res.dump_md(md_writer, f"{nr}md_content.md", image_dir)

        p2 = f"{f_dir}{sep}{nr}draw2.png"
        pip_res.draw_layout(p2)
        pdf_to_png(p2)

        p3 = f"{f_dir}{sep}{nr}draw3.png"
        pip_res.draw_span(p3)
        pdf_to_png(p3)

        p4 = f"{f_dir}{sep}{nr}draw4.png"
        pip_res.draw_line_sort(p4)
        pdf_to_png(p4)
    else:
        print("using cached version")

    if want != "md_content.md":
        return send_file(f_path, as_attachment=True)

    file_paths = [f"{md_image_dir}{sep}{f}" for f in os.listdir(md_image_dir)]
    file_paths.append(f_path)
    memory_file = BytesIO()
    with ZipFile(memory_file, "w") as zf:
        for file_path in file_paths:
            zf.write(file_path, os.path.basename(file_path))
    memory_file.seek(0)
    return Response(
        memory_file.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f"attachment;filename=md_content.zip"},
    )


def pdf_to_png(pdf_path):

    dpi = 150
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    doc.close()
    img.save(pdf_path, "png")


# --- 3. CREATE THE FLASK APP ---
app = Flask(__name__)


@app.route("/", methods=["GET"])
def say_hallo():
    return jsonify({"message": "hallo zakir"})


@app.route("/predict", methods=["POST"])
def predict():
    """The main endpoint for your GUI to call."""
    try:
        image_file = request.files["image"]
        image_bytes = image_file.read()
        print(request.form)
        return detect_layout_miner_u_online(image_bytes, request.form)

    except Exception as e:
        print(traceback.format_exc())
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/advance", methods=["POST"])
def predict_advance():
    """The main endpoint for your GUI to call."""
    try:
        image_file = request.files["image"]
        image_group_bytes: bytes = image_file.read()
        fo = request.form
        fo = json.loads(fo.get("json"))
        print(fo)
        sep = fo.get("seperator").encode("latin")
        idx = fo.get("idx")
        data_sets = []
        lang = "ch_server"
        for im_bytes in image_group_bytes.split(sep):
            if im_bytes:
                data_sets.append(ImageDataset(im_bytes, lang=lang))

        inf_res_list: list[InferenceResult] = batch_doc_analyze(
            data_sets,
            parse_method="ocr",
            lang=lang,
            show_log=True,
        )
        image_dir = "./temp"
        os.makedirs(image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(image_dir)
        final_res = {}

        final_res = {
            "md-content": {},
            "content-list": {},
            "middle-json": {},
            "page-size": {},
        }
        for i, inf_res in enumerate(inf_res_list):
            pip_res: PipeResult = inf_res.pipe_ocr_mode(
                image_writer, lang=lang
            )
            cont = pip_res.get_content_list(image_dir)
            md_cont = pip_res.get_markdown(image_dir)
            midd_json = json.loads(pip_res.get_middle_json())

            para_blocks = midd_json.get("pdf_info", [])[0].get(
                "para_blocks", []
            )
            page_size = midd_json.get("pdf_info", [])[0].get("page_size", [])
            print(page_size)
            id = idx[i]
            final_res["md-content"][id] = md_cont
            final_res["content-list"][id] = cont
            final_res["middle-json"][id] = para_blocks
            final_res["page-size"][id] = page_size

        return jsonify(final_res), 200

    except Exception as e:
        print(traceback.format_exc())
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# This will run the Flask app and keep the notebook cell running.
app.run(port=5000)
