from transformers import AutoTokenizer
from collections import OrderedDict
import torch.nn as nn
from transformers import RemBertConfig, RemBertForSequenceClassification
import torch

def create_model(num_classes):
    model = RemBertForSequenceClassification.from_pretrained("google/rembert", num_labels=3, problem_type="multi_label_classification")
    return model

def setup_model(num_classes):
    model = create_model(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Wrap model with DataParallel
    else:
      global new_state_dict
      new_state_dict = OrderedDict()
      checkpoint = torch.load('tone(1).pth', map_location=torch.device('cpu'))
      for k, v in checkpoint.items():
          name = k[7:] if k.startswith('module.') else k  # Remove `module.` if it exists
          new_state_dict[name] = v
    del checkpoint
    torch.cuda.empty_cache()  # Optional: Clears GPU cache
    model.to(device)
    return model, device




classes={0:'not ignore', 1:'ignore'}


def prepare_input(text, tokenizer):
    encoded_inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask'] 
    
    return input_ids.to(device), attention_mask.to(device)


def predict_ignore(text, model, tokenizer, device):
    classes={0:'-2', 1:'0', 2:'2'}
    input_ids, attention_mask= prepare_input(text, tokenizer)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        del input_ids, attention_mask,model
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        torch.cuda.empty_cache()  
        return (classes[preds.cpu().item()]), torch.sigmoid(logits).cpu().numpy()[0]






model,device=setup_model(2)

model.load_state_dict(new_state_dict)
tokenizer = AutoTokenizer.from_pretrained("tokenizer")


text = """Фото kgd.ru«Наш дом находится на улице Есет батыра, 73 «А». Мы прошли по программе модернизации домов, ремонт должны были делать еще весной, с конца марта, но никаких работ нет! Уже скоро осень наступит. 78 миллионов должны были выделить на ремонт крыши, подвала, фасада, электрики. 56 лет дому, и никогда ремонта не было. Ходили в городской акимат, в СПК, все направляют в областной акимат. Почему? Когда у нас будет ремонт? Мы же сгорим все! У нас уже горела проводка! Это опасно», - жалуются читатели «Диапазону». ""– Дом на ул. Есет батыра проходит по программе модернизации. На сегодняшний день ожидаем финансовые средства из республиканского бюджета. Как поступят денежные средства, сразу начнем ремонтные работы, – ответила Акнур Казбаева [на фото], руководитель отдела жилищной инспекции города Актобе."""""

label, prob=predict_ignore(text, model , tokenizer, device)
print(f'label: {label}; probability: {prob[0]}')

