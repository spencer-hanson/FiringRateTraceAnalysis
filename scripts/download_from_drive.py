import json
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly", "https://www.googleapis.com/auth/drive"]


def get_files(service):
    def get_parents(fid):
        val = service.files().get(fileId=fid, fields="name,parents,id").execute()
        if "parents" in val and val["parents"]:
            val2 = get_parents(val["parents"][0])
            return [val, *val2]
        else:
            return [val]

    if os.path.exists("all_drive_files.json"):
        with open("all_drive_files.json", "r") as f:
            return json.load(f)
    else:
        files_data = []  # List of [[output.hdf id, mousetype, date], ..]
        done = False
        results = service.files().list(includeItemsFromAllDrives=True, supportsAllDrives=True, corpora="allDrives",
                                       q="name = 'output.hdf'").execute()
        count = 0
        while not done:
            for result in results["files"]:
                parent_list = get_parents(result["id"])
                basefile = parent_list[0]["id"]
                mousetype = parent_list[1]["name"]
                date = parent_list[2]["name"]
                print(f"Found output.hdf for {parent_list[2]['name']}")
                files_data.append([basefile, mousetype, date])
            count = count + 1
            if "nextPageToken" in results:
                results = service.files().list(includeItemsFromAllDrives=True, supportsAllDrives=True,
                                               corpora="allDrives", pageToken=results["nextPageToken"],
                                               q="name = 'output.hdf'").execute()
            else:
                done = True

        print("Saving results to file..")
        with open(f"all_outputs{count}.json", "w") as f:
            json.dump(files_data, f)
        return files_data


def download_files(service, listdata, use_cached=True):
    if not os.path.exists("google_drive"):
        os.mkdir("google_drive")

    for filedata in listdata:
        fid, mousetype, date = filedata

        req = service.files().get_media(fileId=fid)
        filename = f"{mousetype}-{date}-output.hdf"
        filepath = f"google_drive/{filename}"
        if os.path.exists(filepath):
            if use_cached:
                print(f"Already downloaded '{filename}..'")
                continue
            else:
                os.remove(filepath)

        print(f"Downloading '{filename}'.. ", end="")
        with open(filepath, "wb") as f:
            done = False
            downloader = MediaIoBaseDownload(f, req)
            while not done:
                status, done = downloader.next_chunk()
                print(f"{int(status.progress() * 100)}% ", end="")
        print("")
    print("Done!")
    tw = 2


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("C:\\Users\\Matrix\\Downloads\\creds.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    files = get_files(service)
    only_mlati = list(filter(lambda x: x[1].startswith("mlati"), files))
    download_files(service, only_mlati, use_cached=True)

    tw = 2


if __name__ == "__main__":
    # os.remove("token.json")
    main()
