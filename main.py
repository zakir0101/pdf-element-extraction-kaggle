import os
import sys
from os.path import sep

# import fitz
# from PIL import Image
from flask import Flask, request, jsonify, send_file, Response
from mineru.cli.common import (
    images_bytes_to_pdf_bytes,
    convert_pdf_bytes_to_bytes_by_pypdfium2,
)
from zipfile import ZipFile
import json
from io import BytesIO
from pyngrok import ngrok
import traceback

# from kaggle_secrets import UserSecretsClient

import copy  # If needed for deep copying data
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import (
    union_make as vlm_union_make,
)
from mineru.utils.enum_class import MakeMode

# from mineru.utils.draw_bbox import draw_layout_bbox # Add if attempting to replicate drawing
# from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2 # Likely not needed for images

from mineru.data.data_reader_writer import (
    FileBasedDataWriter,
)

# from magic_pdf.data.dataset import ImageDataset
# from magic_pdf.model.doc_analyze_by_custom_model import (
#     doc_analyze,
#     batch_doc_analyze,
# )
# from magic_pdf.operators.models import InferenceResult, PipeResult


# --- 1. SETUP THE TUNNEL ---
# Authenticate ngrok using the secret you stored

# user_secrets = UserSecretsClient()
# ngrok.set_auth_token(user_secrets.get_secret("NGROK_AUTH_TOKEN"))
ngrok.set_auth_token(sys.argv[1] if len(sys.argv) > 1 else None)
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
        image_dir_basename = str(os.path.basename(md_image_dir))

        os.makedirs(md_image_dir, exist_ok=True)

        image_writer, md_writer = FileBasedDataWriter(
            md_image_dir
        ), FileBasedDataWriter(f_dir)

        # lang = "ch_server" # VLM models often auto-detect language
        # ds = ImageDataset(img_bytes, lang=lang) # Old dataset creation

        # inf_res: InferenceResult = ds.apply( # Old analysis call
        #     doc_analyze,
        #     ocr=True,
        #     lang=lang,
        #     show_log=True,
        # )

        # p5 = f"{f_dir}{sep}{nr}draw5.png" # Old drawing
        # inf_res.draw_model(p5) # Old drawing
        # pdf_to_png(p5) # Old drawing

        # New VLM analysis
        middle_json, infer_result = vlm_doc_analyze(
            img_bytes, image_writer=image_writer, backend="sglang-engine"
        )
        pdf_info = middle_json["pdf_info"]

        # pip_res: PipeResult = inf_res.pipe_ocr_mode(image_writer, lang=lang) # Old pipe ocr mode

        # Generate MD content using VLM output
        md_content_str = vlm_union_make(
            pdf_info, MakeMode.MM_MD, image_dir_basename
        )
        with open(
            os.path.join(f_dir, f"{nr}md_content.md"), "w", encoding="utf-8"
        ) as f:
            f.write(md_content_str)
        # pip_res.dump_md(md_writer, f"{nr}md_content.md", image_dir_basename) # Old md dump

        p2 = f"{f_dir}{sep}{nr}draw2.png"
        # Try to replicate draw_layout with draw_layout_bbox if available and simple
        # from mineru.utils.draw_bbox import draw_layout_bbox # Ensure this import is active if used
        # draw_layout_bbox(pdf_info, img_bytes, md_image_dir, os.path.basename(p2)) # Example usage
        # For now, commenting out as draw_layout_bbox might need specific setup or parameters
        # pip_res.draw_layout(p2) # Old drawing
        # pdf_to_png(p2) # Old drawing

        # p3 = f"{f_dir}{sep}{nr}draw3.png" # Old drawing
        # pip_res.draw_span(p3) # Old drawing
        # pdf_to_png(p3) # Old drawing

        # p4 = f"{f_dir}{sep}{nr}draw4.png" # Old drawing
        # pip_res.draw_line_sort(p4) # Old drawing
        # pdf_to_png(p4) # Old drawing
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


# def pdf_to_png(pdf_path):
#
#     dpi = 150
#     doc = fitz.open(pdf_path)
#     page = doc.load_page(0)
#     pix = page.get_pixmap(dpi=dpi)
#     mode = "RGBA" if pix.alpha else "RGB"
#     img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
#     doc.close()
#     img.save(pdf_path, "png")


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
        # data_sets = [] # Old way of collecting datasets
        # lang = "ch_server" # VLM models often auto-detect language

        # Define a base temporary directory for images for this batch
        base_temp_image_dir_advance = "./temp_advance_images"
        os.makedirs(base_temp_image_dir_advance, exist_ok=True)
        # Create a unique image directory for each image in the batch if needed by image_writer,
        # or a single one if image_writer handles unique naming.
        # For simplicity, let's use one for the batch, assuming image_writer handles distinct names if it writes anything.
        image_writer_advance = FileBasedDataWriter(base_temp_image_dir_advance)
        image_dir_basename_advance = os.path.basename(
            base_temp_image_dir_advance
        )

        final_res = {
            "md-content": {},
            "content-list": {},
            "middle-json": {},  # This will store para_blocks
            "page-size": {},
        }

        for i, im_bytes in enumerate(image_group_bytes.split(sep)):
            if not im_bytes:
                continue

            # if im_bytes: # Old way of appending to dataset
            # data_sets.append(ImageDataset(im_bytes, lang=lang))

            # New VLM analysis per image

            pdf_bytes = images_bytes_to_pdf_bytes(im_bytes)
            # pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
            middle_json, _ = vlm_doc_analyze(
                pdf_bytes,
                image_writer=image_writer_advance,
                backend="transformers",  # sglang-engine
            )

            # Assuming middle_json["pdf_info"] is a dictionary for a single image.
            # If it can be a list (e.g. if one image could have multiple 'pages' in VLM's view),
            # we might need to adjust. For now, proceed as if it's a dict or the first item of a list.
            pdf_info_list = middle_json.get("pdf_info")
            if isinstance(pdf_info_list, list) and pdf_info_list:
                pdf_info = pdf_info_list[
                    0
                ]  # Taking the first page's info if it's a list
            elif isinstance(
                pdf_info_list, dict
            ):  # It might directly be a dict if not a list
                pdf_info = pdf_info_list
            else:  # Fallback if pdf_info is not found or empty
                pdf_info = {}

            md_cont = vlm_union_make(
                pdf_info, MakeMode.MM_MD, image_dir_basename_advance
            )
            cont = vlm_union_make(
                pdf_info, MakeMode.CONTENT_LIST, image_dir_basename_advance
            )
            para_blocks = pdf_info.get("para_blocks", [])
            page_size = pdf_info.get("page_size", [])

            current_id = idx[i]  # Get the current id using the loop index 'i'

            final_res["md-content"][current_id] = md_cont
            final_res["content-list"][current_id] = cont
            final_res["middle-json"][current_id] = para_blocks
            final_res["page-size"][current_id] = page_size

        # inf_res_list: list[InferenceResult] = batch_doc_analyze( # Old batch analyze call
        #     data_sets,
        #     parse_method="ocr",
        #     lang=lang,
        #     show_log=True,
        # )
        # image_dir = "./temp" # Old image directory
        # os.makedirs(image_dir, exist_ok=True) # Old image directory
        # image_writer = FileBasedDataWriter(image_dir) # Old image writer

        # for i, inf_res in enumerate(inf_res_list): # Old loop through results
        # pip_res: PipeResult = inf_res.pipe_ocr_mode( # Old processing
        #     image_writer, lang=lang
        # )
        # cont = pip_res.get_content_list(image_dir) # Old content list
        # md_cont = pip_res.get_markdown(image_dir) # Old markdown
        # midd_json = json.loads(pip_res.get_middle_json()) # Old middle json

        # para_blocks = midd_json.get("pdf_info", [])[0].get( # Old para_blocks extraction
        #     "para_blocks", []
        # )
        # page_size = midd_json.get("pdf_info", [])[0].get("page_size", []) # Old page_size extraction
        # print(page_size)
        # id = idx[i]
        # final_res["md-content"][id] = md_cont
        # final_res["content-list"][id] = cont
        # final_res["middle-json"][id] = para_blocks
        # final_res["page-size"][id] = page_size

        return jsonify(final_res), 200

    except Exception as e:
        print(traceback.format_exc())
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# This will run the Flask app and keep the notebook cell running.
app.run(port=5000)
