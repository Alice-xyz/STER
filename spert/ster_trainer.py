import argparse
import math
import os

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer
import torch.nn.functional as F

from spert import models
from spert import sampling
from spert import util
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader
from spert.loss import SpERTLoss, Loss, CRDLoss, COSNCELoss
from tqdm import tqdm
from spert.trainer import BaseTrainer


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class STERTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        # args.tokenizer_path，the vocab.txt
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase)

        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader, input_reader_cls_TeaE: BaseInputReader, input_reader_cls_TeaR: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # new add // train_dataset_TeaE, train_dataset_TeaR, validation_dataset_TeaE, validation_dataset_TeaR
        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, args.crd_k, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)
        input_reader_TeaE = input_reader_cls_TeaE(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, args.crd_k, self._logger)
        input_reader_TeaE.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader_TeaE)
        input_reader_TeaR = input_reader_cls_TeaR(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, args.crd_k, self._logger)
        input_reader_TeaR.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader_TeaR)

        train_dataset = input_reader.get_dataset(train_label)
        train_dataset_TeaE = input_reader_TeaE.get_dataset(train_label)
        train_dataset_TeaR = input_reader_TeaR.get_dataset(train_label)
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)
        # validation_dataset_TeaE = input_reader_TeaE.get_dataset(valid_label)
        # validation_dataset_TeaR = input_reader_TeaR.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # add later// model_path_pretrained_TeaE, model_path_pretrained_TeaR
        # new add// model_TeaE, model_TeaR
        # create model
        model_class = models.get_model(self.args.model_type)
        model_class_TeaE = models.get_model(self.args.model_type)
        model_class_TeaR = models.get_model(self.args.model_type)

        # load model
        config = BertConfig.from_pretrained(self.args.model_path)
        config_TeaE = BertConfig.from_pretrained(self.args.model_path_pretrained_TeaE)
        config_TeaR = BertConfig.from_pretrained(self.args.model_path_pretrained_TeaR)
        # config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        util.check_version(config, model_class, self.args.model_path)
        util.check_version(config_TeaE, model_class_TeaE, self.args.model_path_pretrained_TeaE)
        util.check_version(config_TeaR, model_class_TeaR, self.args.model_path_pretrained_TeaR)

        config.spert_version = model_class.VERSION
        config_TeaE.spert_version = model_class_TeaE.VERSION
        config_TeaR.spert_version = model_class_TeaR.VERSION
        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer)
        model_TeaE = model_class_TeaE.from_pretrained(self.args.model_path_pretrained_TeaE,
                                            config=config_TeaE,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader_TeaE.relation_type_count - 1,
                                            entity_types=input_reader_TeaE.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer)
        model_TeaR = model_class_TeaR.from_pretrained(self.args.model_path_pretrained_TeaR,
                                            config=config_TeaR,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader_TeaR.relation_type_count - 1,
                                            entity_types=input_reader_TeaR.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer)

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)
        model_TeaE.to(self._device)
        model_TeaR.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)
        # contrastive_loss = CRDLoss(self.args.train_batch_size, config.hidden_size * 2 + self.args.size_embedding, config.hidden_size * 3 + self.args.size_embedding * 2, 128, self.args.crd_k, self.args.crd_t, self.args.crd_m) #
        contrastive_loss_entity = COSNCELoss()
        contrastive_loss_rel = COSNCELoss()
        contrastive_loss_entity.to(self._device)
        contrastive_loss_rel.to(self._device)
        # mse
        # entity_logits = F.normalize(entity_logits, p=2, dim=-1)
        # entity_logits_TeaE = F.normalize(entity_logits_TeaE, p=2, dim=-1)
        # loss_mse = torch.nn.MSELoss(reduction='mean')  # loss_mse(entity_logits, entity_logits_TeaE)
        # loss = loss_mse(entity_logits, entity_logits_TeaE)
        # CE
        # t = 0.7
        # t_probs = torch.nn.functional.softmax(entity_logits_TeaR/t, dim=-1)
        # loss = loss_ce(entity_logits.view(-1, entity_logits.shape[2])/t, torch.max(t_probs.view(-1, t_probs.shape[2]), dim=-1)[1].long())
        # eval validation set before training
        max_f1_valid = 90
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, model_TeaE, model_TeaR, compute_loss, contrastive_loss_entity, contrastive_loss_rel, optimizer, train_dataset, train_dataset_TeaE, train_dataset_TeaR, updates_epoch, epoch)

            # eval validation sets
            # if not args.final_eval or (epoch == args.epochs - 1):
            macro_f1_rel_nec = self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)
            if macro_f1_rel_nec > max_f1_valid :
                max_f1_valid = macro_f1_rel_nec
                # self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                #                  optimizer=optimizer if self.args.save_optimizer else None,
                #                  include_iteration=False, name='final_model_' + str(epoch))

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        # config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        config = BertConfig.from_pretrained(self.args.model_path)
        util.check_version(config, model_class, self.args.model_path)

        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    # add later// model_TeaE, model_TeaR, dataset_TeaE, dataset_TeaR
    def _train_epoch(self, model: torch.nn.Module, model_TeaE: torch.nn.Module, model_TeaR: torch.nn.Module, compute_loss: Loss, contrastive_loss_entity, contrastive_loss_rel,optimizer: Optimizer, dataset: Dataset, dataset_TeaE: Dataset, dataset_TeaR: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)
        # shuffle set new？？
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=False, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        dataset_TeaE.switch_mode(Dataset.TRAIN_MODE)
        data_loader_TeaE = DataLoader(dataset_TeaE, batch_size=self.args.train_batch_size, shuffle=False, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        dataset_TeaR.switch_mode(Dataset.TRAIN_MODE)
        data_loader_TeaR = DataLoader(dataset_TeaR, batch_size=self.args.train_batch_size, shuffle=False, drop_last=True,
                                      num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch, batch_TeaE, batch_TeaR in tqdm(zip(data_loader, data_loader_TeaE, data_loader_TeaR), total=total, desc='Train epoch %s' % epoch):
            # create data loader, use random seed to get three parts with same order
            # before train, first get the model.eval() representations(including penultimate layer and softmax result: features_TeaE, features_TeaR, wt_TeaE, wt_TeaR) from the pretrained teachers, then
            # entity_features_TeaE, rel_features_TeaE, entity_features_TeaR, rel_features_TeaR, wt_TeaE, wt_TeaR
            with torch.no_grad():
                model_TeaE.eval()
                batch_TeaE = util.to_device(batch_TeaE, self._device)
                # run model (forward pass)
                # entity_logits_TeaE, rel_logits_TeaE, rels_TeaE, entity_features_TeaE, rel_features_TeaE = model_TeaE(encodings=batch_TeaE['encodings'], context_masks=batch_TeaE['context_masks'],
                #                entity_masks=batch_TeaE['entity_masks'], entity_sizes=batch_TeaE['entity_sizes'],
                #                entity_spans=batch_TeaE['entity_spans'], entity_sample_masks=batch_TeaE['entity_sample_masks'],
                #                evaluate=True, pretrained_evaluate=True)
                entity_logits_TeaE, rel_logits_TeaE, entity_features_TeaE, rel_features_TeaE = model_TeaE(encodings=batch_TeaE['encodings'], context_masks=batch_TeaE['context_masks'],
                                              entity_masks=batch_TeaE['entity_masks'], entity_sizes=batch_TeaE['entity_sizes'],
                                              relations=batch_TeaE['rels'], rel_masks=batch_TeaE['rel_masks'])
                model_TeaR.eval()
                batch_TeaR = util.to_device(batch_TeaR, self._device)
                # run model (eval forward pass)
                # entity_logits_TeaR, rel_logits_TeaR, rels_TeaR, entity_features_TeaR, rel_features_TeaR = model_TeaR(encodings=batch_TeaR['encodings'], context_masks=batch_TeaR['context_masks'],
                #                                        entity_masks=batch_TeaR['entity_masks'], entity_sizes=batch_TeaR['entity_sizes'],
                #                                        entity_spans=batch_TeaR['entity_spans'], entity_sample_masks=batch_TeaR['entity_sample_masks'],
                #                                        evaluate=True)
                entity_logits_TeaR, rel_logits_TeaR, entity_features_TeaR, rel_features_TeaR = model_TeaR(encodings=batch_TeaR['encodings'], context_masks=batch_TeaR['context_masks'],
                                              entity_masks=batch_TeaR['entity_masks'], entity_sizes=batch_TeaR['entity_sizes'],
                                              relations=batch_TeaR['rels'], rel_masks=batch_TeaR['rel_masks'])
                # eval the results
            model.train()
            batch = util.to_device(batch, self._device)
            # forward step
            entity_logits, rel_logits, entity_features, rel_features = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'])

            M = entity_features.shape[1] * dataset.document_count
            # add new loss with contrastive learning
            contrastive_entity_loss = contrastive_loss_entity(entity_features, entity_features_TeaE) + contrastive_loss_entity(entity_features, entity_features_TeaR)#idx
            contrastive_rel_loss = contrastive_loss_rel(rel_features, rel_features_TeaE) + contrastive_loss_rel(rel_features, rel_features_TeaR)#idx
            # contrastive_entity_loss, contrastive_rel_loss = contrastive_loss(entity_features, rel_features, entity_features_TeaE, rel_features_TeaE, entity_logits_TeaE, rel_logits_TeaE, entity_features_TeaR, rel_features_TeaR, entity_logits_TeaR, rel_logits_TeaR, batch["crd_samples_idx"], batch["crd_samples_idx"][:, 0], M) #idx
            # compute loss and optimize parameters
            batch_loss = compute_loss.contrastive_compute(entity_logits=entity_logits, rel_logits=rel_logits, contrastive_entity_loss=contrastive_entity_loss, contrastive_rel_loss=contrastive_rel_loss,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks'])

            # batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,
            #                                   rel_types=batch['rel_types'], entity_types=batch['entity_types'],
            #                                   entity_sample_masks=batch['entity_sample_masks'],
            #                                   rel_sample_masks=batch['rel_sample_masks'])

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               evaluate=True)
                entity_clf, rel_clf, rels, entity_features, rel_features = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()
        return rel_nec_eval[-1] # macro_f1_rel_nec

    def _eval_batch(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               evaluate=True)
                entity_clf, rel_clf, rels = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()


    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
