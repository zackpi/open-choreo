# open-choreo
Use openpose to determine how well you match your choreography!


### setup
* go on AWS and check that you have permissions to run a GPU instance.
  * My Service Quotas -> EC2 -> type "p2" or "p3"
* if not, click on a pN.8xlarge instance and request quota increase
  * hopefully they approve in 30 min, then 30 min after that you can run one
* launch a p2 (K80 gpu) or p3 (v100 gpu) instance
  * go to homepage -> EC2 -> Launch Instance
  * select Deep Learning AMI (Ubuntu) Version 24.1 -> pN.8xlarge -> Review and Launch
  * create a new key pair, download it -> launch

