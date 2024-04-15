import argparse
import json
import os
import time
from itertools import chain

from question_answering.clevr_generation import (
    instantiate_templates_dfs,
    parser,
)


def side_inputs_extractor(program):
    return tuple(
        chain(
            *[
                node["side_inputs"] if "side_inputs" in node else []
                for node in program
            ]
        )
    )


def main(args):
    with open(args.metadata_file, "r") as f:
        metadata = json.load(f)
        dataset = metadata["dataset"]
        if dataset != "CLEVR-v1.0" and dataset != "DATASETNAME-v1.0":
            raise ValueError('Unrecognized dataset "%s"' % dataset)

    functions_by_name = {}
    for f in metadata["functions"]:
        functions_by_name[f["name"]] = f
    metadata["_functions_by_name"] = functions_by_name

    # Load templates from disk
    # Key is (filename, file_idx)
    num_loaded_templates = 0
    templates = {}
    for fn in os.listdir(args.template_dir):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(args.template_dir, fn), "r") as f:
                base = os.path.splitext(fn)[0]
                for i, template in enumerate(json.load(f)):
                    num_loaded_templates += 1
                    key = (fn, i)
                    templates[key] = template
        except Exception as e:
            print(f"Error loading template {fn}: {e}")
    print("Read %d templates from disk" % num_loaded_templates)

    def reset_counts():
        # Maps a template (filename, index) to the number of questions we have
        # so far using that template
        template_counts = {}
        # Maps a template (filename, index) to a dict mapping the answer to the
        # number of questions so far of that template type with that answer
        template_answer_counts = {}
        node_type_to_dtype = {
            n["name"]: n["output"] for n in metadata["functions"]
        }
        for key, template in templates.items():
            template_counts[key[:2]] = 0
            final_node_type = template["nodes"][-1]["type"]
            final_dtype = node_type_to_dtype[final_node_type]
            answers = metadata["types"][final_dtype]
            if final_dtype == "Bool":
                answers = [True, False]
            if final_dtype == "Integer":
                if metadata["dataset"] == "CLEVR-v1.0":
                    answers = list(range(0, 11))
                else:
                    answers = list(range(0, 100))  # not tested
            template_answer_counts[key[:2]] = {}
            for a in answers:
                template_answer_counts[key[:2]][a] = 0
        return template_counts, template_answer_counts

    template_counts, template_answer_counts = reset_counts()

    # Read file containing input scenes
    all_scenes = []
    with open(args.input_scene_file, "r") as f:
        scene_data = json.load(f)
        all_scenes = scene_data["scenes"]
        scene_info = scene_data["info"]
    begin = args.scene_start_idx
    if args.num_scenes > 0:
        end = args.scene_start_idx + args.num_scenes
        all_scenes = all_scenes[begin:end]
    else:
        all_scenes = all_scenes[begin:]

    # Read synonyms file
    with open(args.synonyms_json, "r") as f:
        synonyms = json.load(f)

    questions = []
    scene_count = 0

    # Heuristic to generate interesting questions:
    # Only allow questions that have more than 1 answer in the training set

    template_counts, template_answer_counts = reset_counts()
    allowed_questions = set()
    for (fn, idx), template in templates.items():
        if args.verbose:
            print("trying template ", fn, idx)
        question2answer = {}
        for i, scene in enumerate(all_scenes[:10]):
            scene_fn = (
                scene["image_filename"]
                if "image_filename" in scene
                else "house_%d" % i
            )
            scene_struct = scene
            print(
                "starting image %s (%d / %d)"
                % (scene_fn, i + 1, len(all_scenes))
            )

            ts, qs, ans = instantiate_templates_dfs(
                scene_struct,
                template,
                metadata,
                template_answer_counts[(fn, idx)],
                synonyms,
                forbid_key_value=json.load(open(args.forbid_key_value_json))
                if args.forbid_key_value_json
                else None,
                max_instances=None,  # generate all instances
                verbose=False,
            )
            image_index = int(os.path.splitext(scene_fn)[0].split("_")[-1])
            for t, q, a in zip(ts, qs, ans):
                question_feature = (fn, idx, side_inputs_extractor(q))
                questions.append(
                    {
                        "split": scene_info["split"],
                        "image_filename": scene_fn,
                        "image_index": image_index,
                        "image": os.path.splitext(scene_fn)[0],
                        "question": t,
                        "program": q,
                        "answer": a,
                        "template_filename": fn,
                        "question_family_index": idx,
                        "question_index": len(questions),
                        "question_feature": question_feature,
                    }
                )
                if (
                    question_feature in question2answer
                    and question2answer[question_feature] != a
                ):
                    allowed_questions.add(question_feature)
                else:
                    question2answer[question_feature] = a
            if args.verbose:
                print(
                    f"Finished scene {scene_fn} and found {len(allowed_questions)} valid questions"
                )
    output_questions = []
    for q in questions:
        if q["question_feature"] in allowed_questions:
            output_questions.append(q)
    questions = output_questions
    # Change "side_inputs" to "value_inputs" in all functions of all functional
    # programs. My original name for these was "side_inputs" but I decided to
    # change the name to "value_inputs" for the public CLEVR release. I should
    # probably go through all question generation code and templates and rename,
    # but that could be tricky and take a while, so instead I'll just do it here.
    # To further complicate things, originally functions without value inputs did
    # not have a "side_inputs" field at all, and I'm pretty sure this fact is used
    # in some of the code above; however in the public CLEVR release all functions
    # have a "value_inputs" field, and it's an empty list for functions that take
    # no value inputs. Again this should probably be refactored, but the quick and
    # dirty solution is to keep the code above as-is, but here make "value_inputs"
    # an empty list for those functions that do not have "side_inputs". Gross.
    for q in questions:
        for f in q["program"]:
            if "side_inputs" in f:
                f["value_inputs"] = f["side_inputs"]
                del f["side_inputs"]
            else:
                f["value_inputs"] = []

    with open(args.output_questions_file, "w") as f:
        print("Writing output to %s" % args.output_questions_file)
        json.dump(
            {
                "info": scene_info,
                "questions": questions,
            },
            f,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.profile:
        import cProfile

        cProfile.run("main(args)")
    else:
        main(args)
