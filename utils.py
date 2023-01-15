import time, os


def linear_learning_rate_scheduler(optimizer, steps, target, warm_up, decay):
    """
    Change the learning rate of the optimizer using a linear learning rate schedule
    :param optimizer: optimizer that are being used
    :param steps: current number of steps
    :param target: maximum learning rate
    :param warm_up: number of warm up steps
    :param decay: number of steps at which the learning rate will be 0
    :return: modified optimizer
    """
    if steps < warm_up:
        running_lr = target * steps / warm_up
    else:
        if decay != 0:
            running_lr = target * (decay - steps) / (decay - warm_up)
        else:
            running_lr = target

    for g in optimizer.param_groups:
        g['lr'] = running_lr
    return optimizer, running_lr


def evaluate(model, train_dataset, dev_dataset, writer, epoch, args):
    model.train(False)
    time_qry_vectors = time.time()
    c_descs_vectors = model.encode_code_synonyms(train_dataset, batch_size=200) # c_desc
    print('code_descriptions: time: %.6f' % (time.time() - time_qry_vectors))

    # rank and eval train
    time_rank = time.time()
    train_metrics = model.rank(
        dataset=train_dataset,
        c_descs_vectors=c_descs_vectors,
        return_eval=True,
        save_topk=(args.sample_method == 'ance'))
    print('rank time: %.6f' % (time.time() - time_rank))

    if args.save_checkpoints and epoch != 0:
        checkpoint_dir = os.path.join('checkpoints', args.version_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

    for key, value in train_metrics[0].items():
        writer.add_scalar(key + '_train', value, epoch)
    writer.flush()

    # rank and eval dev
    time_rank = time.time()
    dev_metrics = model.rank(
        dataset=dev_dataset,
        c_descs_vectors=c_descs_vectors,
        return_eval=True,
        save_topk=False)
    print('rank time: %.6f' % (time.time() - time_rank))

    for key, value in dev_metrics[0].items():
        writer.add_scalar(key + '_dev', value, epoch)
    writer.flush()

    model.train(True)


def get_datasets(
    config_name,
    model_name,
    version='mimic3',
    truncate_length=4000,
    label_truncate_length=50,
    term_count=4,
    sample_method='pos_neg',
    n_ranks_ance=100,
    n_samples=(20, 20),
    return_tensors="np",
    loss='cross_entropy',
    switch_qd=False,
    chunk_notes=False,
    ):

    sort_method='random'


    from data_util import MimicFullDataset
    train_dataset = MimicFullDataset(
        config_name=config_name,
        version=version,
        mode="train",
        model_name=model_name,
        truncate_length=truncate_length,
        label_truncate_length=label_truncate_length,
        term_count=term_count,
        sort_method=sort_method,
        sample_method=sample_method,
        n_ranks_ance=n_ranks_ance,
        n_samples=n_samples,
        return_tensors=return_tensors,
        loss=loss,
        switch_qd=switch_qd,
        chunk_notes=chunk_notes
    )

    dev_dataset = MimicFullDataset(
        config_name=config_name,
        version=version,
        mode="dev",
        model_name=model_name,
        truncate_length=truncate_length,
        label_truncate_length=label_truncate_length,
        term_count=term_count,
        sort_method=sort_method,
        sample_method=sample_method,
        n_ranks_ance=n_ranks_ance,
        n_samples=n_samples,
        return_tensors=return_tensors,
        loss=loss,
        switch_qd=switch_qd,
        chunk_notes=chunk_notes
    )
    test_dataset = MimicFullDataset(
        config_name=config_name,
        version=version,
        mode="test",
        model_name=model_name,
        truncate_length=truncate_length,
        label_truncate_length=label_truncate_length,
        term_count=term_count,
        sort_method=sort_method,
        sample_method=sample_method,
        n_ranks_ance=n_ranks_ance,
        n_samples=n_samples,
        return_tensors=return_tensors,
        loss=loss,
        switch_qd=switch_qd,
        chunk_notes=chunk_notes
    )

    dev_dataset.c_desc_input_ids = train_dataset.c_desc_input_ids
    dev_dataset.c_desc_attention_mask = train_dataset.c_desc_attention_mask
    dev_dataset.c_desc_filter_indices = train_dataset.c_desc_filter_indices
    test_dataset.c_desc_input_ids = train_dataset.c_desc_input_ids
    test_dataset.c_desc_attention_mask = train_dataset.c_desc_attention_mask
    test_dataset.c_desc_filter_indices = train_dataset.c_desc_filter_indices

    return train_dataset, dev_dataset, test_dataset