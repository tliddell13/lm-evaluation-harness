import collections
import itertools
import random

import lm_eval.permutations as p
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests
from lm_eval.models.gpt2 import HFLM

import numpy as np
import transformers

CoT_Prompt = (
"Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\\n"
"A: Let's think step by step\\n\\n"
"There are 15 trees originally.\\n\\n"
"Then there were 21 trees after some more were planted.\\n\\n"
"So there must have been 21 - 15 = 6.\\n\\n"  
"The answer is 6.\\n\\n"
"#### 6 \\n\\n"
"Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\\n"
"A: Let's think step by step\\n\\n"
"There are originally 3 cars.\\n\\n"
"2 more cars arrive.\\n\\n"
"3 + 2 = 5.\\n\\n"
"The answer is 5.\\n\\n"
"#### 5 \\n\\n"
"Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\\n"
"A: Let's think step by step\\n\\n"
"Originally, Leah had 32 chocolates.\\n\\n"
"Her sister had 42.\\n\\n"
"So in total they had 32 + 42 = 74.\\n\\n"
"After eating 35, they had 74 - 35 = 39.\\n\\n"
"The answer is 39.\\n\\n"
"#### 39 \\n\\n"
"Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\\n"
"A: Let's think step by step\\n\\n"
"Jason started with 20 lollipops.\\n\\n"
"Then he had 12 after giving some to Denny.\\n\\n"
"So he gave Denny 20 - 12 = 8.\\n\\n"
"The answer is 8.\\n\\n"
"#### 8 \\n\\n"
"Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\\n"
"A: Let's think step by step\\n\\n"
"Shawn started with 5 toys.\\n\\n"
"If he got 2 toys each from his mom and dad, then that is 4 more toys.\\n\\n"
"5 + 4 = 9.\\n\\n"
"The answer is 9.\\n\\n"
"#### 9 \\n\\n"
"Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\\n"
"A: Let's think step by step\\n\\n"
"There were originally 9 computers.\\n\\n"
"For each of 4 days, 5 more computers were added.\\n\\n"
"So 5 * 4 = 20 computers were added.\\n\\n"
"9 + 20 is 29.\\n\\n"
"The answer is 29.\\n\\n"
"#### 29 \\n\\n"
"Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\\n"
"A: Let's think step by step\\n\\n"
"Michael started with 58 golf balls.\\n\\n"
"After losing 23 on tuesday, he had 58 - 23 = 35.\\n\\n"
"After losing 2 more, he had 35 - 2 = 33 golf balls.\\n\\n"
"The answer is 33.\\n\\n" 
"#### 33 \\n\\n" 
"Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\\n"
"A: Let's think step by step\\n\\n"
"Olivia had 23 dollars.\\n\\n"
"5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.\\n\\n"
"So she has 23 - 15 dollars left.\\n\\n"
"23 - 15 is 8.\\n\\n"
"The answer is 8."
"#### 8 \\n\\n"
)

@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    max_batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    shuffle=None,
    shuffleAnswer=None,
    remove_question=False,
    posReplace=None,
    extra_answers=False,
    named_entities=None,
    cot = False,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model, transformers.PreTrainedModel object, or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write details about prompts and logits to json for all tasks
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir.
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    elif isinstance(model, transformers.PreTrainedModel):
        lm = lm_eval.models.get_model("hf-causal")(
            pretrained=model,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )
        no_cache = True
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    if not no_cache:
        lm = lm_eval.base.CachingLM(
            lm,
            "lm_cache/"
            + (model if isinstance(model, str) else model.model.config._name_or_path)
            + "_"
            + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
            + ".db",
        )

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        output_base_path=output_base_path,
        shuffle = shuffle,
        shuffleAnswer = shuffleAnswer,
        remove_question = remove_question,
        posReplace = posReplace,
        extra_answers = extra_answers,
        named_entities = named_entities,
        cot = cot
    )

    # add info about the model and few shot config
    model_name = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, transformers.PreTrainedModel):
        model_name = "pretrained=" + model.config._name_or_path
    results["config"] = {
        "model": model_name,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "batch_sizes": list(lm.batch_sizes.values())
        if hasattr(lm, "batch_sizes")
        else [],
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "description_dict": description_dict,
    }

    return results


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    provide_description=None,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    # Custom argument to shuffle the word order in the question
    shuffle=None,
    shuffleAnswer=None,
    remove_question=False,
    posReplace=None,
    extra_answers=False,
    named_entities=None,
    cot = False,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param write_out: bool
        If True, write all prompts, logits and metrics to json for offline analysis
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir
    :param shuffle: str, optional
        shuffles the question using either trigram, bigram, or unigram. Defaults to none.
    :param shuffleAnswer: str, optional
        shuffles the answer using either trigram, bigram, or unigram. Defaults to none.
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print(
            "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
        )

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}
    write_out_info = {}

    docs_for_decontamination = collections.defaultdict(list)

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        if write_out:
            prompt_details = []

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )
        if limit is not None:
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        # Load the model if we are appending extra answers
        if extra_answers:
            tokenizer, model = p.load_model("/users/adbt150/archive/Llama-2-7b-hf")
       
        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )
            # Generate an extra answer using the subject
            """"
            word, pos = p.get_sentence_subject(doc['query'])
            sentence = p.generate_fake_answer(word, pos, model, tokenizer)
            # Add the generated sentence to the doc
            doc['choices'].append(sentence)
            """
            # If question is accessed using 'query'
            if 'query' in doc:
                # Check if shuffle is set
                if shuffle:
                    doc['query'] = p.shuffle(doc['query'], task_name, shuffle)
                # Check if PosReplace is set
                if posReplace == "VERB":
                    doc['query'] = p.verbSynonyms(doc['query'])
                # Check if remove question is set
                if remove_question:
                    doc['query'] = ' '
                if named_entities == "remove":
                    doc['query'] = p.removeNE(doc['query'])
                if named_entities == "keep":
                    doc['query'] = p.onlyNE(doc['query'])
                if extra_answers:
                    word, pos = p.get_sentence_subject(doc['query'])
                    sentence = p.generate_fake_answer(word, pos, model, tokenizer)
                    # Add the generated sentence to the doc
                    doc['choices'].append(sentence)
            # If question is accessed using 'sentence'
            elif 'sentence' in doc:
                if shuffle:
                    doc['sentence'] = p.shuffle(doc['sentence'], task_name, shuffle)
                if posReplace == "VERB":
                    doc['sentence'] = p.verbSynonyms(doc['sentence']) 
                if remove_question:
                    doc['sentence'] = ' '
                if named_entities == "remove":
                    doc['sentence'] = p.removeNE(doc['sentence'])
                if named_entities == "keep":
                    doc['sentence'] = p.onlyNE(doc['sentence'])
            # If question is accessed using 'question'
            elif 'question' in doc:
                if shuffle:
                    doc['question'] = p.shuffle(doc['question'], task_name, shuffle)
                if posReplace == "VERB":
                    doc['question'] = p.verbSynonyms(doc['question'])
                if remove_question:
                    doc['question'] = ' '
                if named_entities == "remove":
                    doc['question'] = p.removeNE(doc['question'])
                if named_entities == "keep":
                    doc['question'] = p.onlyNE(doc['question'])
    
                if extra_answers: 
                    print("Extra answers")
                    # Generate an extra answer using the subject
                    word, pos = p.get_sentence_subject(doc['question'])
                    sentence = p.generate_fake_answer(word, pos, model, tokenizer)
                    # Add the generated sentence to the doc
                    doc['choices'].append(sentence)
            
            if cot and task_name == "gsm8k":
                doc["question"] = CoT_Prompt + "\n\n" + "Question: " + doc["question"]
                    
            docs[(task_name, doc_id)] = doc
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
            )
            reqs = task.construct_requests(doc, ctx)

            if write_out:
                prompt_details.append({"doc_id": doc_id})

            # print the prompt for the first few documents
            if doc_id < 2:
                print(
                    f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                )
                print("Requests:", reqs)
          
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

                if write_out:
                    prompt_details[-1][f"prompt_{i}"] = "".join(
                        (map(lambda x: "".join(x), req.args))
                    )

        if write_out:
            write_out_info[task_name] = prompt_details

    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit
        )

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

            if write_out:
                write_out_info[task_name][doc_id][f"logit_{i}"] = resp
                task = task_dict[task_name]
                if isinstance(task, lm_eval.base.MultipleChoiceTask):
                    write_out_info[task_name][doc_id]["truth"] = doc["gold"]
                elif isinstance(task, lm_eval.tasks.winogrande.Winogrande):
                    write_out_info[task_name][doc_id]["truth"] = task.answer_to_num[
                        doc["answer"]
                    ]
                else:
                    write_out_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)

    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, requests)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

            if write_out:
                write_out_info[task_name][doc_id][metric] = str(value)

            # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
            if decontaminate and task_name in overlaps:
                if doc_id not in overlaps[task_name]:
                    vals[(task_name, metric + decontaminate_suffix)].append(value)

    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        real_metric = metric  # key when looking up the metric with task.aggregation
        if metric.endswith(decontaminate_suffix):
            real_metric = metric.replace(
                decontaminate_suffix, ""
            )  # decontaminated still uses the same metric
        results[task_name][metric] = task.aggregation()[real_metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[real_metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )

        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)

    if write_out:
        import json
        import pathlib

        output_base_path = (
            pathlib.Path(output_base_path)
            if output_base_path is not None
            else pathlib.Path(".")
        )
        try:
            output_base_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        for task_name, _ in task_dict_items:
            # change name of write out based on custom arguments
            # If question shuffle
            if shuffle == "unigram":
                with open(
                    output_base_path.joinpath("question_shuffle_" + f"{task_name}_write_out_info.json"),
                    "w",
                    encoding="utf8",
                ) as fp:
                    json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)
            # If answer shuffle
            elif shuffleAnswer == "unigram":
                with open(
                    output_base_path.joinpath("answer_shuffle_" +f"{task_name}_write_out_info.json"),
                    "w",
                    encoding="utf8",
                ) as fp:
                    json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)
            # If question and answer shuffle
            elif shuffle == "unigram" and shuffleAnswer == "unigram":
                with open(
                    output_base_path.joinpath("QandA_shuffle_" + f"{task_name}_write_out_info.json"),
                    "w",
                    encoding="utf8",
                ) as fp:
                    json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)
            # If question removed
            elif remove_question:
                with open(
                    output_base_path.joinpath("removeQ_" + f"{task_name}_write_out_info.json"),
                    "w",
                    encoding="utf8",
                ) as fp:
                    json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)
            # If verb replace
            elif posReplace == "VERB":
                with open(
                    output_base_path.joinpath("VERB_replace" + f"{task_name}_write_out_info.json"),
                    "w",
                    encoding="utf8",
                ) as fp:
                    json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)
            elif named_entities == "remove" or named_entities == "keep":
                with open(
                    output_base_path.joinpath(f"{named_entities}_NE_" + f"{task_name}_write_out_info.json"),
                    "w",
                    encoding="utf8",
                ) as fp:
                    json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)
            # None of the above
            else:
                with open(
                    output_base_path.joinpath(f"{task_name}_write_out_info.json"),
                    "w",
                    encoding="utf8",
                ) as fp:
                    json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)

    return {"results": dict(results), "versions": dict(versions)}

# make a row with the average of hendrycks test's acc_norm
def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()
