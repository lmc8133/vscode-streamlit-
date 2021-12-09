import math
import torch
import json
import numpy as np
import time
from PIL import Image

import clip

import sys
sys.path.append('../code/phraseloceval/lib/')
from phraseloc.eval.dataset import Dataset, DatasetLoader


# ========= box searching strategy =======
class BoxProposal(object):

    def __init__(self, alpha):
        
        self.alpha = alpha

    def shrink_strategy(self):

        choices = ['shrink_to_left', 'shrink_to_right',
                  'shrink_to_up', 'shrink_to_bottom']

        def box_transform(bbox, c):
            x1, y1, x2, y2 = bbox.tolist()
            w, h = x2 - x1, y2 - y1

            new_bbox = list()
            if c == 'shrink_to_left':  # shrink_to_left
                new_bbox = [x1, y1, x2 - self.alpha * w, y2]
            elif c == 'shrink_to_right':  # shrink_to_right
                new_bbox = [x1 + self.alpha * w, y1, x2, y2]
            elif c == 'shrink_to_up':  # shrink_to_up
                new_bbox = [x1, y1, x2, y2 - self.alpha * h]
            else:  # shrink_to_bottom
                new_bbox = [x1, y1 + self.alpha * h, x2, y2]

            return np.array(new_bbox).astype(bbox.dtype)

        return choices, box_transform
# ========================================


class CLIPBeamGrounding(object):

    def __init__(self, model_type, device, alpha=0.2):

        self.device = device

        model, preprocess = clip.load(model_type, device=device)

        self.model = model
        self.img_preprocess = preprocess
        self.txt_tokenize = clip.tokenize
        choices, box_transform = BoxProposal(alpha).shrink_strategy()
        self.choices = choices
        print('[box choices]:', choices)
        self.box_transform = box_transform

    @torch.no_grad()
    def search(self, img_path, text_query, beam_size=3, patience=2, max_steps=20, threshold=0.5):

        beam_size = max(1, min(beam_size, len(self.choices)))

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        query_emb = self.encode_query(text_query)  # (1, 1024)
        logit_scale = self.model.logit_scale.exp()

        cur_bboxes = list()  # [bbox0, bbox1, ...]
        histories = list()  # [[(bbox, score), (bbox, score), ...], ...]
        finished = list()
        counters = list() # [(highest_score, patience_counter), ... ] && len(records) == len(cur_bboxes)

        for step in range(max_steps):
            # print(cur_bboxes)
            # propose next bounding box
            if step == 0:
                bbox_proposals = self.propose_next_bboxes(np.array([0, 0, w, h]), w, h)
                bbox_embeddings = self.encode_bboxes(img, bbox_proposals)

                dist = (logit_scale * bbox_embeddings @ query_emb.T).squeeze(dim=-1)  # (len(Actions),)
                # print(dist)
                _, inds = dist.topk(beam_size)
                # print(_)
                # print(inds)
                cur_bboxes = [bbox_proposals[i].tolist() for i in inds.tolist()]
                counters = [(dist[i].item(), 0) for i in inds.tolist()]
                histories = [[(np.array([0, 0, w, h]), 0.), (bbox_proposals[i], dist[i].item())] for i in inds.tolist()]

            else:
                all_bbox_proposals = list()
                for bbox in cur_bboxes:
                    all_bbox_proposals.extend(self.propose_next_bboxes(np.array(bbox), w, h))
                bbox_embeddings = self.encode_bboxes(img, all_bbox_proposals)

                dist = (logit_scale * bbox_embeddings @ query_emb.T).squeeze(dim=-1)  # (len(Actions) * beam_size,)
                _, inds = dist.sort(descending=True)

                next_bboxes = list()
                next_counters = list()
                updated_histories = list()
                for i in inds.tolist():

                    if len(next_bboxes) + len(finished) == beam_size:
                        break

                    oi = i // len(self.choices)  # ind used in ongoing list

                    highest_score, counter = counters[oi]

                    if dist[i] < highest_score + threshold and counter == patience:
                        finished.append(histories[oi].copy())
                    else:
                        box_proposal = all_bbox_proposals[i].tolist()
                        if box_proposal in next_bboxes:
                            continue

                        next_bboxes.append(box_proposal)
                        updated_histories.append(histories[oi].copy())
                        updated_histories[-1].append((all_bbox_proposals[i], dist[i].item()))

                        if dist[i] >= highest_score + threshold:
                            next_counters.append((dist[i], 0))
                        else:
                            next_counters.append((highest_score, counter + 1))

                cur_bboxes = next_bboxes
                histories = updated_histories
                counters = next_counters

            if len(cur_bboxes) == 0:
                break

            # sanity check
            assert len(cur_bboxes) == len(histories)
            assert (len(histories) + len(finished)) == beam_size

        # 剪裁 patience 阶段次优的搜索结果
        results = list()
        for h in finished:
            if patience == 0:
                results.append(h)
            else:
                results.append(h[:-1*patience])

        for h, (_, counter) in zip(histories, counters):
            if counter > 0:
                results.append(h[:-1*counter])
            else:
                results.append(h)

        best_choice = sorted(list(range(beam_size)), key=lambda k : results[k][-1][1], reverse=True)[0]

        history = [h[0] for h in results[best_choice]]
        clip_scores = [h[1] for h in results[best_choice]]

        return history, clip_scores

    def propose_next_bboxes(self, cur_bbox, width, height):
        """

        :param cur_bbox: numpy.array (x1, y1, x2, y2)
        :param width: width of the original image
        :param height: float of the original image
        :return: bbox_proposals (List of numpy.array)
        """

        bbox_proposals = list()
        for c in self.choices:
            new_bbox = self.box_transform(cur_bbox, c)
            bbox_proposals.append(new_bbox)

        return bbox_proposals

    @torch.no_grad()
    def encode_bboxes(self, img, bbox_list):
        """

        :param img: PIL.Image
        :param bbox_list: List of numpy.array
        :return: feature (n, 512)
        """

        cropped_imgs = list()
        for box in bbox_list:
            x = img.copy().crop(box.tolist())
            cropped_imgs.append(self.img_preprocess(x))

        input_ = torch.tensor(np.stack(cropped_imgs)).to(self.device)
        bbox_embeddings = self.model.encode_image(input_)
        bbox_embeddings /= bbox_embeddings.norm(dim=-1, keepdim=True)

        return bbox_embeddings

    @torch.no_grad()
    def encode_query(self, text_query):

        input_ = self.txt_tokenize(text_query).to(self.device)
        # print(input_)
        text_embeddings = self.model.encode_text(input_)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

        return text_embeddings


#***************************************************
def quanti_eval(cbg, json_path, coco_path):     #定量评测函数，陆明聪，11.30 15：13
    loader=DatasetLoader()
    dataset=loader.read_json(json_path)
    img_path=dataset.imagename
    final_img_path=[coco_path+path for path in img_path]
    query=dataset.get_phraselist()
    # print(final_img_path[:20])
    # print(query[:20])
    # print(len(query))
    # print(len(final_img_path))
    beam_pred=[]
    # s=time.time()
    for i,qu in enumerate(query):
        print(i,'/',len(query))
        img_p=final_img_path[i]
        qu_text=' '.join(qu)
        history, clip_scores = cbg.search(img_p, qu_text, beam_size=3, patience=2, max_steps=20, threshold=0.5)
        final_box=history[-1]       #此处得到的box表示为[x1,y1,x2,y2]
        comparable_box=final_box.copy()
        comparable_box[2]=comparable_box[2]-comparable_box[0]
        comparable_box[3]=comparable_box[3]-comparable_box[1]
        #表示为[x1,y1,w,h]
        t=comparable_box.tolist()
        beam_pred.append({"box":t, "image":"", "entity":"", "phraseId":"", 
        "imageName":"", "phrase":[], "categories":[]})
        if i<1000:
            # print(img_p)
            # print(qu_text)
            # print(history)
            pass
        else:
            # print(type(comparable_box))
            # print(beam_pred)
            break
    with open('beam_pred_v2_refcoco+_1000_a2b3p2t.5.json','w', encoding='utf-8') as f:
        json.dump(beam_pred,f)
        print('finish')
    # e=time.time()
    # print(e-s)


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model = "RN50"
    clip_model = "ViT-B/32"
    cbg = CLIPBeamGrounding(clip_model, device)


    refcoco_path='../code/phraseloceval/data/refcoco/annotation_test.json'
    refcoco_plus_path='../code/phraseloceval/data/refcoco+/annotation_test.json'
    refcocog_path='../code/phraseloceval/data/refcocog/annotation_test.json'
    cocoimg_path='../../../mnt/ssd_01/data_base/MS_COCO/coco/images/train2014/'
    quanti_eval(cbg,refcoco_plus_path,cocoimg_path)

    # pic_path = '/data/kebobei/1128/tmp/man-dog-sea.jpg'
    # pic_path = '/data/kebobei/1128/tmp/418.jpg'
    # query = 'blue shirt'
    # query = 'man'
    # query = 'man'
    # query = 'beach beach beach beach beach'
    # query = 'a man with blue shirt and brown shorts'
    # query = 'dog'
    # query = 'a man and a dog'
    # query = 'blue sky'
    # query = 'sea'
    # history, clip_scores = cbg.search(pic_path, query, beam_size=3, patience=1, max_steps=20, threshold=0.5)

    # print('='*20)
    # print('history')
    # print(history)
    # print('clip_scores')
    # print(clip_scores)
