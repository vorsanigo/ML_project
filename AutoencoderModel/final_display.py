

def create_final_dict(res_dict):

    '''This function returns the final dictionary with the group name and a dictionary of results'''

    final_res = dict()
    final_res["groupname"] = "Innominati"
    final_res["images"] = res_dict
    return final_res


def create_results_dict(results_dict, query_img, gallery_list):

    '''This function returns the dictionary of the results.
    Specifically, the query and the rispective gallery images ranked'''

    results_dict[query_img] = []
    for gallery_img in gallery_list:
        results_dict[query_img].append(gallery_img)



