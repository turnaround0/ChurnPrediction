def apply_to_users(users, posts):
    users['numPosts'] = posts.groupby('OwnerUserId').size()
    return users


def apply_to_posts(users, posts):
    return posts
