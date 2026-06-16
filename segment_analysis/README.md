# Segment analysis commentary

This is a narrative commentary on my process analyzing segments of the Peakbagger GPX data.

## Roadblocks

A few things came up that required cleaning. First, when I initially plotted slope against speed, I got this:

<img width="1980" height="1350" alt="image" src="https://github.com/user-attachments/assets/98b81267-1974-4ea6-b241-bb0e13c7f250" />

The image banding is not natural. I am using linear interpolation to generate these 40m segments, and this revealed that several of my GPX tracks 
both recorded time at steady 1 minute intervals and also contained numerous repeated points at the same time stamp. The combined effect was
lots of 40m segments that 'started' and 'ended' exactly 1, or 2 minutes apart. 40 meters divided by 60 seconds produces a MPH of 1.491, which occurs 
way more than it should here.

To remedy this, I just removed points with repeated time stamps. Now, I get a more natural plot, without horizontal banding:

<img width="1980" height="1350" alt="image" src="https://github.com/user-attachments/assets/ea435900-eed2-48be-bd4f-66fc9a72cfe2" />

But there were still segments that didn't make sense. Namely, a bunch of points that were both above 10 mph and 20 degrees of absolute slope.
To be exact -- 89 such points when using those exact cutoffs. And they all belonged to the same GPS track! Rendezvous Mountain, 08/02/2025... 
they were on a ski lift. After removing this track, I finally got this distribution:

<img width="1980" height="1350" alt="image" src="https://github.com/user-attachments/assets/d648cc07-2ba8-453d-9643-c999806e8fc1" />

Which reflects pretty closely what I'd expect.
