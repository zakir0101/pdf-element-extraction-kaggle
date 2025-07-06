import copy
import json
import os
import sys
import traceback
from io import BytesIO
from os.path import sep
from pprint import pprint
from zipfile import ZipFile

from flask import Flask, Response, jsonify, request, send_file
from mineru.backend.pipeline.model_json_to_middle_json import \
    result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import \
    doc_analyze as pipeline_doc_analyze
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import \
    union_make as vlm_union_make
from mineru.cli.common import (convert_pdf_bytes_to_bytes_by_pypdfium2,
                               images_bytes_to_pdf_bytes)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from pyngrok import ngrok

ngrok.set_auth_token(sys.argv[1] if len(sys.argv) > 1 else os.environ['NGROK_KEY'])
public_url = ngrok.connect(5000)
print(f"✅ Kaggle is now live at: {public_url}")
print(public_url)


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
        return "hallo Zakir ", 200
        # detect_layout_miner_u_online(image_bytes, request.form)

    except Exception as e:
        print(traceback.format_exc())
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


def proccess_pdf_info(pdf_info_list, index):

    global image_dir_basename_advance, file_names, final_res

    if isinstance(pdf_info_list, list) and pdf_info_list:
        pdf_info = pdf_info_list[0]
    elif isinstance(pdf_info_list, dict):
        pdf_info = pdf_info_list
    else:
        pdf_info = {}

    md_cont = vlm_union_make(
        pdf_info_list, MakeMode.MM_MD, image_dir_basename_advance
    )
    cont = vlm_union_make(
        pdf_info_list,
        MakeMode.CONTENT_LIST,
        image_dir_basename_advance,
    )
    para_blocks = pdf_info.get("para_blocks", [])
    page_size = pdf_info.get("page_size", [])

    current_id = file_names[index]

    final_res["md-content"][current_id] = md_cont
    final_res["content-list"][current_id] = cont
    final_res["middle-json"][current_id] = para_blocks
    final_res["page-size"][current_id] = page_size


@app.route("/predict/advance", methods=["POST"])
def predict_advance():
    """The main endpoint for your GUI to call."""
    try:
        image_file = request.files["image"]
        image_group_bytes: bytes = image_file.read()
        fo = request.form
        fo = json.loads(fo.get("json"))
        print(fo)

        global image_dir_basename_advance, file_names, final_res

        sep = fo.get("seperator").encode("latin")
        file_names = fo.get("idx")
        model = fo.get("mode")
        # data_sets = [] # Old way of collecting datasets
        # lang = "ch_server" # VLM models often auto-detect language

        # Define a base temporary directory for images for this batch
        base_temp_image_dir_advance = "./temp_advance_images"
        os.makedirs(base_temp_image_dir_advance, exist_ok=True)

        image_writer_advance = FileBasedDataWriter(base_temp_image_dir_advance)
        image_dir_basename_advance = os.path.basename(
            base_temp_image_dir_advance
        )

        final_res = {
            "md-content": {},
            "content-list": {},
            "middle-json": {},
            "page-size": {},
        }

        print("\n**********************************************************")
        print("********************* Start Processing *********************\n")
        if model == "pipeline":
            print("\n**************** [ mode == Pipeline ] ******************")

            pdf_bytes_list = []
            for i, im_bytes in enumerate(image_group_bytes.split(sep)):
                if not im_bytes:
                    continue
                pdf_bytes_list.append(images_bytes_to_pdf_bytes(im_bytes))

            (
                infer_results,
                all_image_lists,
                all_pdf_docs,
                lang_list,
                ocr_enabled_list,
            ) = pipeline_doc_analyze(
                pdf_bytes_list,
                ["ch_server"] * len(pdf_bytes_list),
                parse_method="ocr",
                formula_enable=True,
                table_enable=True,
            )

            for idx, model_list in enumerate(infer_results):
                model_json = copy.deepcopy(model_list)
                images_list = all_image_lists[idx]
                pdf_doc = all_pdf_docs[idx]
                _lang = "ch_server"
                _ocr_enable = True
                p_formula_enable = True
                middle_json = pipeline_result_to_middle_json(
                    model_list,
                    images_list,
                    pdf_doc,
                    image_writer_advance,
                    _lang,
                    _ocr_enable,
                    p_formula_enable,
                )

                pdf_info_list = middle_json["pdf_info"]
                proccess_pdf_info(pdf_info_list, idx)

        else:

            print("\n**************** [ mode == VLM ] ******************")
            for i, im_bytes in enumerate(image_group_bytes.split(sep)):
                if not im_bytes:
                    continue

                pdf_bytes = images_bytes_to_pdf_bytes(im_bytes)

                middle_json, _ = vlm_doc_analyze(
                    pdf_bytes,
                    image_writer=image_writer_advance,
                    backend=model,  # sglang-engine
                )

                pdf_info_list = middle_json.get("pdf_info")

                proccess_pdf_info(pdf_info_list, i)

        print("\n***********************  [DONE]  ***********************\n")

        return jsonify(final_res), 200

    except Exception as e:
        print(traceback.format_exc())
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


# This will run the Flask app and keep the notebook cell running.
app.run(port=5000)
