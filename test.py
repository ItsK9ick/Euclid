import torch
from torch.utils.data import DataLoader
import Dbd_Model

if __name__ == '__main__':
    dataset_test_0 = ["E:/temp/dbd/test/0"]
    dataset_test_1 = ["E:/temp/dbd/test/1"]
    labels = [0, 1]
    checkpoint = "./lightning_logs/version_1/checkpoints/epoch=6-step=5754.ckpt"

    test_data = Dbd_Model.CustomDataset(dataset_test_0 + dataset_test_1, labels, transform=Network.transforms_test)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # my_model = Network.MyModel()
    my_model = Dbd_Model.My_Model.load_from_checkpoint(checkpoint)

    for sample in test_dataloader.dataset:
        pred = my_model(sample[0].unsqueeze(0))
        y_true = sample[1]
        y_pred = torch.argmax(pred, -1)
        print("pred: {}, true: {}".format(y_pred.item(), y_true))

    # for img in train_features:
    #     img_np = torch.permute(img, [1, 2, 0]).numpy()
    #     img_np = (img_np * 255).astype(np.uint8)
    #     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("", img_np)
    #     cv2.waitKey(0)
