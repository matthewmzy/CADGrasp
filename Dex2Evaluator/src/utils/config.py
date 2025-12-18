import yaml
from os.path import dirname, join

# use this so that one can use config.x.y.z instead of config['x']['y']['z']
class DotDict(dict):
    def __getattr__(self, item):
        if item in self.keys():
            return self[item]
        return None
    
    def __setattr__(self, key, value):
        self[key] = value

def to_dot_dict(dic):
    for k in dic.keys():
        if type(dic[k]) == dict:
            dic[k] = to_dot_dict(dic[k])
    return DotDict(dic)

def to_dict(args):
    result = dict()
    for k, v in args.items():
        if isinstance(v, dict):
            result[k] = to_dict(v)
        else:
            result[k] = v
    return result

def add_argparse(parser, arg_mapping):
    for raw_key, (_, arg_type, default) in arg_mapping:
        parser.add_argument('--' + raw_key, type=arg_type, default=default)
    return parser

# combine config from yaml file and argument
# priority: args in console > default args > yaml file
def load_config(yaml_file, arg_mapping=None, args=None):
    with open(yaml_file, 'r') as f:
        dic = yaml.load(f, Loader=yaml.FullLoader)
    if args is not None:
        for raw_key, (new_key, _, _) in arg_mapping:
            value = eval(f'args.{raw_key}')
            if value is None:
                continue
            temp = dic
            for k in new_key.split('/')[:-1]:
                if not k in temp.keys():
                    temp[k] = dict()
                elif not type(temp[k]) == dict:
                    raise ValueError
                temp = temp[k]
            temp[new_key.split('/')[-1]] = value
    return to_dot_dict(dic)

def ckpt_to_config(ckpt_path):
    config_path = join(dirname(dirname(ckpt_path)), 'config.yaml')
    return load_config(config_path)